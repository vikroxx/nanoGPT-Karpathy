# Implementing GPT based on Attention is All You Need using Pytorch

Pytorch implementation of **GPT-2** (125M) , thanks to the OG Andrej Karpathy's Youtube series. Implemented GPT-2 from scratch using pytorch, along with the training loop, logging on WandB and successful model trainig on [Openwebtext-10k dataset](https://huggingface.co/datasets/stas/openwebtext-10k).

## Requirments:
Ensure you have Python 3.9 or higher installed. Install the required packages using pip:

`pip install -r requirements.txt`

## Model Summary
### Transformer Architecture
The model implements a GPT-2-like architecture using PyTorch, based on the "Attention is All You Need" paper. The configuration allows for adjusting various parameters such as the number of layers, heads, embedding dimensions, and more.

### Layer-by-Layer Implementation for Forward Pass:
- Embedding Layers: **Token** and **positional** embeddings provide input representations, which are summed and passed through a dropout layer.

- *Transformer Blocks*:
Each block consists of:
    - A **multi-head** self-attention mechanism.
    - **Layer normalization** followed by a residual connection.
    - A **feed-forward** neural network with another layer normalization and a second residual connection.
- *Output Processing*: The final output of the transformer blocks goes through another layer normalization.
- *Language Model head* : Finally after layer normalization, a linear layer for the final logits.

### Optimizer
- The training uses the AdamW optimizer, configured with a weight decay for regularization. 
- It differentiates between parameters by applying weight decay only to those with two or more dimensions, leaving biases and other parameters without decay.

## Training Process and Results

### Training Loop
The training process involves:
- Dynamic learning rate adjustments with a warm-up phase followed by cosine decay.
- Gradient accumulation to effectively handle large batch sizes that might not fit into memory all at once.
- Regular evaluation of both training and validation losses to monitor progress and overfitting.
- Usage of Weights & Biases (WandB) for logging training metrics in real-time.

### WandB Logging
Training metrics such as loss and learning rate are logged to WandB, providing a visual and historical track of training progress. This allows for monitoring model performance and adjusting training parameters as needed.

## Conclusion
- The model was trained on a **Vast.ai instance** equipped with an **RTX 4090** for **400 epochs**. It achieved a **training loss** of **0.15**, indicating a state of being fully overfitted to the training dataset. 
- This demonstrates the model's capacity to learn a specific dataset thoroughly but also highlights the need for strategies to mitigate overfitting for generalization to unseen data.

![Overfitting Model on Dataset](https://i.ibb.co/DW6wZ1j/train-loss.png)


## Acknowledgments
This implementation of **GPT-2** was inspired by and based in part on the educational content provided by Andrej Karpathy through his [YouTube series on deep learning](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). 
The design and structure of the model, along with the training techniques, draw upon concepts and code examples from his [nanoGPT Repo](https://github.com/karpathy/nanoGPT).
