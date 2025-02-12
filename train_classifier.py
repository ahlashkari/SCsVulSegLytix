"""
Trains BERT on BCCC-VulSCs-2024.
"""


import os
import warnings
from math import ceil
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import BertConfig, GPT2TokenizerFast

from bert_hf import BertForSequenceClassification


def get_config() -> dict:
    virtual_batch_size = 2_048
    actual_batch_size = 8 * torch.cuda.device_count()
    config = {
        'path': 'orig/',
        'device': 'cuda',
        'compile': True,
        'num_workers': 4 * torch.cuda.device_count(),
        'vocab_size': 50_304,
        'seq_len': 8_192,
        'num_epochs': 20,
        'virtual_batch_size': virtual_batch_size,
        'actual_batch_size': actual_batch_size,
        'batches_to_accum': virtual_batch_size // actual_batch_size,
        'layers': 2,
        'emb_dim': 128,
        'hidden_dim': 128,
        'heads': 4,
        'hidden_dropout': 0.25,
        'atten_dropout': 0.0,
        'mlp_head': False,
        'lr': 3e-3,
        'wd': 1e-1
    }
    return config


def get_data(
    path: str,
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv('labels.csv')
    vul_cols = ['CallToUnknown', 'DenialOfService', 'IntegerUO', 'Reentrancy']
    input_paths = path + df['ID'] + '.sol'
    target = df[vul_cols]
    return (input_paths[df['is_validation'] == False],
            input_paths[df['is_validation'] == True],
            target[df['is_validation'] == False],
            target[df['is_validation'] == True])


def train_tokenizer(path: str, vocab_size: int) -> GPT2TokenizerFast:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                  special_tokens=['[PAD]', '[CLS]'])
    tokenizer.train(list(map(lambda p: str(p), Path(path).glob('*'))),
                    trainer=trainer)

    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    return GPT2TokenizerFast(tokenizer_object=tokenizer)


class SCDataset(Dataset):
    def __init__(
        self,
        input: pd.Series,
        target: pd.DataFrame,
        tokenizer: GPT2TokenizerFast,
        seq_len: int,
        ) -> None:
        self.input = input
        self.target = torch.tensor(target.values, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(
        self,
        ind: int,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with open(self.input.iloc[ind]) as file:
            input = self.tokenizer(file.read(), return_tensors='pt',
                                   padding='max_length', truncation=True,
                                   max_length=self.seq_len-1)
        return (torch.cat((torch.tensor([1]), input['input_ids'].squeeze(0))),
                torch.cat((torch.tensor([1]), input['attention_mask'].squeeze(0))),
                self.target[ind, :])


def train_model(
    model: BertForSequenceClassification,
    train_dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: BCEWithLogitsLoss,
    device: torch.device,
    batches_to_accum: int,
    lr_scheduler: OneCycleLR,
    num_epochs: int,
    ) -> None:
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        optimizer.zero_grad()

        for ind, (input, mask, target) in enumerate(train_dl):
            input, mask, target = input.to(device), mask.to(device), target.to(device)

            with autocast('cuda'):
                output = model(input, mask).logits
                loss = loss_fn(output, target)

            scaler.scale(loss).backward()

            if ((ind + 1) % batches_to_accum == 0) or ((ind + 1) == len(train_dl)):
                print(f'Iter {ind+1}/{len(train_dl)}')
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

        print(f'Epoch {epoch + 1} completed')


def validate_model(
    model: BertForSequenceClassification,
    valid_dl: DataLoader,
    device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    outputs = []
    targets = []
    with torch.no_grad():
        for input, mask, target in valid_dl:
            input, mask = input.to(device), mask.to(device)
            output = 0.0 < model(input, mask).logits
            outputs.append(output.detach().cpu().numpy())
            targets.append(target.numpy())

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    return outputs, targets


def main() -> None:
    warnings.filterwarnings('ignore')
    config = get_config()
    device = config['device']

    tokenizer = train_tokenizer(config['path'], config['vocab_size'])
    tokenizer.save_pretrained('tokenizer')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    input_train, input_valid, target_train, target_valid = get_data(config['path'])
    train_ds = SCDataset(input_train, target_train, tokenizer, config['seq_len'])
    valid_ds = SCDataset(input_valid, target_valid, tokenizer, config['seq_len'])
    train_dl = DataLoader(train_ds, batch_size=config['actual_batch_size'],
                          shuffle=True, num_workers=config['num_workers'],
                          pin_memory=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=config['actual_batch_size'],
                          num_workers=config['num_workers'], pin_memory=True,
                          drop_last=True)

    model_config = BertConfig(vocab_size=train_ds.tokenizer.vocab_size,
                              max_position_embeddings=config['seq_len'] + 64,
                              num_hidden_layers=config['layers'],
                              hidden_size=config['emb_dim'],
                              num_attention_heads=config['heads'],
                              intermediate_size=config['hidden_dim'],
                              layer_norm_eps=1e-5,
                              problem_type='multi_label_classification',
                              hidden_dropout_prob=config['hidden_dropout'],
                              attention_probs_dropout_prob=config['atten_dropout'],
                              num_labels=4)
    model = (BertForSequenceClassification(model_config, mlp_head=config['mlp_head'])
             .to(device))

    optimizer = AdamW(model.parameters(), lr=config['lr'],
                      weight_decay=config['wd'])
    scaler = GradScaler()
    lr_scheduler = OneCycleLR(optimizer, max_lr=config['lr'], epochs=config['num_epochs'],
                              steps_per_epoch=ceil(len(train_dl) / config['batches_to_accum']))

    pos_weight = torch.tensor([(len(train_ds) + len(valid_ds) - s) / s
                               for s in [11_131, 12_394, 16_740, 17_698]])
    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    if config['compile']:
        model = torch.compile(model)

    train_model(model, train_dl, optimizer, scaler, loss_fn, device,
                config['batches_to_accum'], lr_scheduler, config['num_epochs'])
    outputs, targets = validate_model(model, valid_dl, device)

    for ind in range(4):
        print(f'Vul: {target_train.columns[ind]}', end=' ')
        print(f'Acc: {round(accuracy_score(targets[:, ind], outputs[:, ind]), 2)}', end=' ')
        print(f'Rec: {round(recall_score(targets[:, ind], outputs[:, ind]), 2)}', end=' ')
        print(f'Prec: {round(precision_score(targets[:, ind], outputs[:, ind]), 2)}', end=' ')
        print(f'F1: {round(f1_score(targets[:, ind], outputs[:, ind]), 2)}')

    model.save_pretrained('classifier')


if __name__ == "__main__":
    main()
