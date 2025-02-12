"""
Extracts the vulnerable and secure segments of smart contracts contained in a folder.
"""

import argparse
import torch
import os
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer
from pathlib import Path

from bert import BertForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Generator adapted from https://github.com/hila-chefer/Transformer-Explainability

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def generate_LRP(self, input_ids, attention_mask,
                     index=None, start_layer=11):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {'alpha': 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cams = []
        blocks = self.model.bert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        rollout[:, 0, 0] = rollout[:, 0].min()
        return rollout[:, 0]


    def generate_LRP_last_layer(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {'alpha': 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn_cam()[0]
        cam = cam.clamp(min=0).mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0]

    def generate_full_lrp(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {'alpha': 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        cam = self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)
        cam = cam.sum(dim=2)
        cam[:, 0] = 0
        return cam

    def generate_attn_last_layer(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
        cam = cam.mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0]

    def generate_rollout(self, input_ids, attention_mask, start_layer=0, index=None):
        self.model.zero_grad()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        blocks = self.model.bert.encoder.layer
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attention.self.get_attn()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        rollout[:, 0, 0] = 0
        return rollout[:, 0]

    def generate_attn_gradcam(self, input_ids, attention_mask, index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {'alpha': 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()
        grad = self.model.bert.encoder.layer[-1].attention.self.get_attn_gradients()

        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam[:, 0, 0] = 0
        return cam[:, 0]


# End of generator


def convert_linear_model(lin: torch.nn.Linear, i: int):
    weight = lin.weight.data[i, :].unsqueeze(0)
    bias = lin.bias.data[i].unsqueeze(0)
    new_weight = torch.concat([torch.zeros_like(weight), weight])
    new_bias = torch.concat([torch.zeros_like(bias), bias])
    new = torch.nn.Linear(new_weight.shape[1], new_weight.shape[0], device=device).eval()
    new.weight.data = new_weight
    new.bias.data = new_bias
    return new


def extract(
    model: BertForSequenceClassification,
    explanations: Generator,
    tokenizer: AutoTokenizer,
    input: str,
    q: float = 0.80,
    ) -> str:
    classifications = ['NEGATIVE', 'POSITIVE']
    encoding = tokenizer([input], return_tensors='pt',
                         truncation=True, max_length=8_192)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)


    logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    output = torch.nn.functional.softmax(logits, dim=-1)
    classification = output.argmax(dim=-1).item()
    class_name = classifications[classification]

    if class_name == 'NEGATIVE':
        return '', input

    expl = explanations.generate_LRP(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     start_layer=0)[0]
    expl = (expl-expl.min()) / (expl.max()-expl.min())

    vul_tokens = input_ids[(torch.quantile(expl, q) < expl).unsqueeze(0)]
    sec_tokens = input_ids[~(torch.quantile(expl, q) < expl).unsqueeze(0)]

    vul_seg_ = tokenizer.convert_ids_to_tokens(vul_tokens)
    sec_seg_ = tokenizer.convert_ids_to_tokens(sec_tokens)
    return ''.join(vul_seg_).replace('Ġ', ' '), ''.join(sec_seg_).replace('Ġ', ' ')


def create_dataset(
    directory: str,
    tokenizer: AutoTokenizer,
    model: BertForSequenceClassification,
    classifier: torch.nn.Linear,
    explanations: Generator,
    vul_ind: int,
    vul_name: str,
    ) -> None:
    model.classifier = convert_linear_model(classifier, vul_ind)

    paths = Path(directory).glob('*.sol')
    os.mkdir(f'{vul_name}/')
    os.mkdir(f'{vul_name}/vul/')
    os.mkdir(f'{vul_name}/sec/')

    for i, p in enumerate(paths):
        if (i+1) % 250 == 0:
            print(i, end='\r', flush=True)

        with open(p) as file:
            input = file.read()

        p = str(p).split('/')[-1]
        vul, sec = extract(model, explanations, tokenizer, input)
        with open(f'{vul_name}/vul/{p}', 'w+') as file:
            file.write(vul)
        with open(f'{vul_name}/sec/{p}', 'w+') as file:
            file.write(sec)


def main():
    parser = argparse.ArgumentParser(description="Extracts the vulnerable and secure segments of smart contracts contained in a folder.")
    parser.add_argument("directory", type=str, help="Path to the directory")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('tokenizer')
    model = BertForSequenceClassification.from_pretrained('classifier').eval().to(device)
    classifier = model.classifier
    explanations = Generator(model)

    vul_cols = ['CallToUnknown', 'DenialOfService', 'IntegerUO', 'Reentrancy']
    for vul_ind, vul_name in zip([0, 1, 2, 3], vul_cols):
        create_dataset(args.directory, tokenizer, model, classifier, explanations,
                       vul_ind, vul_name)


if __name__ == '__main__':
    main()
