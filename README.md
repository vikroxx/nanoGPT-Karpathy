# Implementing GPT based on Attention is All You Need using Pytorch

Pytorch implementation of **GPT-2** (125M) , thanks to the OG Andrej Karpathy's Youtube series. Implemented GPT-2 from scratch using pytorch, along with the training loop, logging on WandB and successful model trainig on [Openwebtext-10k dataset](https://huggingface.co/datasets/stas/openwebtext-10k).

![Overfitting Model on Dataset](https://i.ibb.co/G3QHwwm/figure-1.png)

## Requirments:
Ensure you have Python 3.9 or higher installed. Install the required packages using pip:

`pip install -r requirements.txt`


## An Introduction to Transformers
1. X and y (inputs and targets) have dimensions of (batch_size, block_size).

Here each batch of X contains list of indexes with length of list as the context size/block size.

Eg. tokenized sentence -> [My,name,is,jasdeep] => Looking up index in the vocabulary => [1000,1050,4,23]

2. Y contains the subsequent output [name,is,jasdeep,.] => Looking up index in the vocabulary => [1050,4,23,8]

The first layer X passes through is the embedding layer(vocab_size,embeddding_dimension). The embedding layer takes in the corresponding indices, plucks the embedding of an index and spits out. The transformation is from

