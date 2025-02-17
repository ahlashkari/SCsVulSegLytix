![](https://github.com/ahlashkari/SCsVulSegLyzer/blob/main/bccc.jpg)

# SCsVulSegLyzer
SCsVulSegLyzer is a learning-based framework for detecting and extracting vulnerable segments in smart contracts (SCs). It leverages a Transformer model - namely, Bidirectional Encoder Representations from Transformers (BERT) - trained with contract-level labels to extract vulnerable and secure segments from contracts. Thanks to its novel use of a post-hoc interpretability technique, it highlights vulnerable segments without the need for expensive line-level annotations during training, and it improves on graph-based methods by avoiding their costly pre-processing phase. Covering a broad range of SC vulnerabilities, SCsVulSegLyzer outperforms prior methods in terms of accuracy and computational complexity. Its goal is to aid developers and security auditors in accurately analyzing the security of SCs by providing a fine-grained view of vulnerability locations without sacrificing efficiency or ease of use.

![Diagram of model.](./model.svg)

In Python-like pseudo-code, SCsVulSegLyzer behaves roughly as follows during inference:

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
rel_scores = trans_att(bert, contract)
# Tokens with a low relevance score are secure, and those with a high score are vulnerable.
# The threshold is calculated by taking the 80th percentile of the scores.
sec_tokens = tokens[quantile(rel_scores, 0.8) > rel_scores]
vul_tokens = tokens[quantile(rel_scores, 0.8) < rel_scores]
# Secure and vulnerable segments are decoded into plain text and returned.
sec = tokenizer.decode(sec_tokens)
vul = tokenizer.decode(vul_tokens)
return sec, vul
```

This process is repeated for each vulnerability type supported by SCsVulSegLyzer; namely, CallToUnknown, DenialOfService, IntegerUO, and Reentrancy.

# Installation

To use SCsVulSegLyzer, please first clone this repository using ```git clone https://github.com/ahlashkari/SCsVulSegLyzer.git```, navigate into it, and install its dependencies via ```pip install -r requirements.txt```. It is recommended to use a virtual environment such as ```venv``` to avoid dependency conflicts.

# Training

Having downloaded BCCC-VulSCs-2024, place the folder containing the SCs in this directory and rename it to ```orig/```. Training can be done by simply executing ```python train_classifier.py```. This saves the tokenizer and BERT model in the folders ```tokenizer/``` and ```classifier/```, respectively. The trained tokenizer and BERT model already ship with this project for convenience.

# Inference

Provided with a directory, the script ```extraction.py``` analyzes the SCs therein and extracts their vulnerable and secure segments. Specifically, it creates four new folders, ```CallToUnknown/```, ```DenialOfService/```, ```IntegerUO/```, and ```Reentrancy/```, each with two subfolders ```sec/``` and ```vul/```, which contain the secure and vulnerable segments of each contract for every vulnerability type. For example, given a directory ```example/``` containing two SCs, ```contract1.sol``` and ```contract2.sol```, running ```python extraction.py example/``` will result in the following structure:

```
[Root]
│── example/
│ ├── contract1.sol
| ├── contract1.sol
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

Here, for a vulnerability named ```Vulnerability```, ```Vulnerability/vul/contract1.sol``` contains the vulnerable segments, if any, in contract ```contract1.sol``` for that vulnerability type, and ```Vulnerability/sec/contract1.sol``` its secure segments. The same applies to ```contract2.sol```. Note that, if no vulnerabilities are found in a contract, its vulnerable segments will be empty. On GPU-enabled machines, this script automatically utilizes CUDA to speed up computations; otherwise, it resorts to the CPU.

# Project Team members

* [**Arash Habibi Lashkari:**](http://ahlashkari.com/index.asp) Founder and supervisor

* [**Borna Ahmadzadeh:**](https://github.com/BobMcDear) Undergraduate student, Researcher and developer - York University ( 6 months years, 2024 - 2024)


# Acknowledgment

This project has been made possible through funding from the Natural Sciences and Engineering Research Council of Canada — NSERC (#RGPIN-2020-04701) and Canada Research Chair (Tier II) - (#CRC-2021-00340) to Arash Habibi Lashkari.
