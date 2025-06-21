![](https://github.com/ahlashkari/SCsVulSegLyzer/blob/main/bccc.jpg)

# SCsVulSegLytix
SCsVulSegLytix is a learning-based, analytics framework for detecting and extracting vulnerable segments in smart contracts (SCs). It leverages a Transformer model - namely, Bidirectional Encoder Representations from Transformers (BERT) - trained with contract-level labels to extract vulnerable and secure segments from contracts. Thanks to its novel use of a post-hoc interpretability technique, it highlights vulnerable segments without the need for expensive line-level annotations during training. It also improves graph-based methods by avoiding their costly pre-processing phase. Covering a broad range of SC vulnerabilities, SCsVulSegLytix outperforms prior methods regarding accuracy and computational complexity. Its goal is to aid developers and security auditors in accurately analyzing the security of SCs by providing a fine-grained view of vulnerability locations without sacrificing efficiency or ease of use.

![Diagram of model.](./model.svg)

In Python-like pseudo-code, SCsVulSegLytix behaves roughly as follows during inference:

```python
"""
Inputs:
    - contract: SC source code as a string.
Outputs:
    - sec: Secure segments in contract.
    - vul: Vulnerable segments in contract, if any.
"""
# Data pre-processing.
tokens = tokenizer.encode(contract)
tokens = tokens[:8_192]

# Contract-level prediction.
prediction = transformer.predict(tokens)

# If the contract was labeled secure, it has no vulnerable segments.
if prediction == 'Negative':
    sec = contract
    vul = ''
    return sec, vul

# Otherwise, its vulnerable segments are extracted.
# The first step is to calculate relevance scores using the
# post-hoc interpretability technique TransAtt, which assigns
# scores to input tokens based on how much they contributed
# to the model's classification decision.
rel_scores = trans_att(bert, tokens)

# Tokens with a low relevance score are secure, and those with a high score are vulnerable.
# The threshold is calculated by taking the 80th percentile of the scores.
sec_tokens = tokens[quantile(rel_scores, 0.8) > rel_scores]
vul_tokens = tokens[quantile(rel_scores, 0.8) < rel_scores]

# Secure and vulnerable segments are decoded into plain text and returned.
sec = tokenizer.decode(sec_tokens)
vul = tokenizer.decode(vul_tokens)

return sec, vul
```

This process is repeated for each vulnerability type supported by SCsVulSegLytix: CallToUnknown, DenialOfService, IntegerUO, and Reentrancy.

# Installation

SCsVulSegLytix is implemented in Python, so please first ensure Python >= 3.8 is installed on your system. Then, clone this repository using ```git clone https://github.com/ahlashkari/SCsVulSegLyzer.git```, navigate into it, and install its dependencies via ```pip install -r requirements.txt```. Using a virtual environment such as ```venv``` is recommended to avoid dependency conflicts. This software has been tested on Ubuntu 22.04 with a V100 GPU and an Intel i7-10700K CPU.

# Training & Evaluation

Having downloaded BCCC-VulSCs-2024, place the folder containing the SCs in this directory, rename it to ```orig/```, and execute ```python -m src.train_classifier```. This trains SCsVulSegLytix on BCCC-VulSCs-2024 for vulnerability classification and extraction, subsequently validating its results on the test set and printing the relevant metrics. After training, the tokenizer and BERT model are saved in the folders ```tokenizer/``` and ```classifier/```, respectively. The trained tokenizer and BERT model have already been shipped with this project for convenience. Training is only supported on devices with Nvidia GPUs; however, inference, described in more depth in the next section, works on all devices.

# Inference

Provided with a directory, the script ```extraction.py``` analyzes the SCs therein and extracts their vulnerable and secure segments. Specifically, it creates four new folders, ```CallToUnknown/```, ```DenialOfService/```, ```IntegerUO/```, and ```Reentrancy/```, each with two subfolders ```sec/``` and ```vul/```, which contain the secure and vulnerable segments of each contract for every vulnerability type. For example, given a directory ```example/``` containing two SCs, ```contract1.sol``` and ```contract2.sol```, running ```python -m src.extraction example/``` will result in the following structure:

```
[Root]
│── example/
│ ├── contract1.sol
| ├── contract2.sol
│── CallToUnknown/
│ ├── sec/
│ │ ├── contract1.sol
│ │ ├── contract2.sol
│ ├── vul/
│ │ ├── contract1.sol
│ │ ├── contract2.sol
│── DenialOfService/
│ ├── sec/
│ │ ├── contract1.sol
│ │ ├── contract2.sol
│ ├── vul/
│ │ ├── contract1.sol
│ │ ├── contract2.sol
│── IntegerUO/
│ ├── sec/
│ │ ├── contract1.sol
│ │ ├── contract2.sol
│ ├── vul/
│ │ ├── contract1.sol
│ │ ├── contract2.sol
│── Reentrancy/
│ ├── sec/
│ │ ├── contract1.sol
│ │ ├── contract2.sol
│ ├── vul/
│ │ ├── contract1.sol
│ │ ├── contract2.sol
...
```

Here, for a vulnerability named ```Vulnerability```, ```Vulnerability/vul/contract1.sol``` contains the vulnerable segments, if any, in contract ```contract1.sol``` for that vulnerability type, and ```Vulnerability/sec/contract1.sol``` its secure segments. The same applies to ```contract2.sol```. Note that if no vulnerabilities are found in a contract, its vulnerable segments will be empty. On GPU-enabled machines, this script automatically utilizes CUDA to speed up computations; otherwise, it resorts to the CPU.

# Performance Metrics

Below are the accuracy, precision, recall, and F1 score of SCsVulSegLytix on the test set for vulnerability classification per vulnerability type:

| Vulnerability     | Accuracy | Recall | Precision | F1 Score |
|------------------|----------|--------|-----------|----------|
| CallToUnknown   | 0.89     | 0.67   | 0.69      | 0.68     |
| DenialOfService | 0.96     | 0.90   | 0.96      | 0.93     |
| IntegerUO       | 0.82     | 0.75   | 0.72      | 0.73     |
| Re-entrancy     | 0.90     | 0.77   | 0.81      | 0.79     |

As for vulnerability extraction, our model's baseline, vulnerified, and securified scores can be seen in the following table. For the interpretation of these scores, please refer to our article.

| Vulnerability    | Vulnerified (↑) | Baseline | Securified (↓) |
|-----------------|----------------|----------|----------------|
| CallToUnknown   | 0.71 (+0.07)    | 0.64     | 0.60 (-0.04)   |
| DenialOfService | 0.95 (+0.04)    | 0.91     | 0.42 (-0.49)   |
| IntegerUO       | 0.72 (+0.04)    | 0.68     | 0.60 (-0.08)   |
| Re-entrancy     | 0.78 (+0.05)    | 0.73     | 0.65 (-0.08)   |

# Copyright (c) 2025

For citation in your works and also understanding SCsVulSegLytix completely, you can find below published papers:

- “SCsVulSegLytix: Detecting and Extracting Vulnerable Segments from Smart Contracts Using Weakly-Supervised Learning”, Borna Ahmadzadeh, Arousha Haghighian Roudsari, Sepideh HajiHosseinKhani and Arash Habibi Lashkari, Journal of Systems and Software,
Volume 231, 2025.

# Project Team members

* [**Arash Habibi Lashkari:**](http://ahlashkari.com/index.asp) Founder and supervisor

* [**Borna Ahmadzadeh:**](https://github.com/BobMcDear) Undergraduate student, Researcher and developer - York University ( 6 months, 2024 - 2024)


# Acknowledgment

This project was made possible by funding from the Natural Sciences and Engineering Research Council of Canada (NSERC - #RGPIN-2020-04701) and Canada Research Chair (Tier II - #CRC-2021-00340) to Arash Habibi Lashkari.
