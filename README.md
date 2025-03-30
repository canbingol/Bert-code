# BERT from Scratch

This repository contains an implementation of the BERT (Bidirectional Encoder Representations from Transformers) model built from scratch using PyTorch. This implementation follows the architecture described in the original BERT paper, including multi-head self-attention mechanisms, feed-forward networks, and the masked language modeling (MLM) pre-training objective.

## Model Architecture

The model follows the original BERT architecture with:

- 12 transformer layers
- 12 attention heads
- 768 hidden dimensions
- 30,522 vocabulary size (matching bert-base-uncased)
- GELU activation functions
- Layer normalization
- Residual connections

## Files

- `bert_model.py`: Contains the implementation of the BERT model architecture, including the attention mechanism, feed-forward network, and layer normalization.
- `bert_dataset_preproces.py`: Handles data preprocessing, tokenization, and dataset creation for the masked language modeling task.
- `train_bert.py`: Contains the training loop, loss calculation, and visualization functions.

## Pre-training Approach

The model is pre-trained using the Masked Language Modeling (MLM) objective:

- 15% of tokens are randomly masked
- Of these masked tokens:
  - 80% are replaced with the [MASK] token
  - 10% are replaced with random tokens
  - 10% remain unchanged
- The model is trained to predict the original tokens for all masked positions

## Usage

### Requirements

- PyTorch
- Transformers (for tokenizer)
- Matplotlib (for visualizations)
- dotenv (for environment variables)

### Training

1. Place your training text data in a file named `data.txt`
2. Set your Hugging Face token in an environment variable:
```
HF_TOKEN=your_token_here
```
3. Run the training script:
```
python train_bert.py
```

## Configuration

The model configuration is defined in `bert_model.py` and includes:

```python
bertconfig = {
    'vocab_size': 30522,  # same with bert-base-uncased
    'd_model': 768,
    'max_len': 256,
    'n_head': 12,
    'n_layer': 12,
    'dff': 4*768,
    'batch_size': 1,
    'device': 'cuda'
}
```

You can adjust these parameters to experiment with different model sizes and configurations.

## Future Improvements

- Increase batch size for more efficient training
- Implement gradient accumulation for memory efficiency
- Add support for Next Sentence Prediction (NSP) task
- Implement fine-tuning capabilities for downstream tasks
- Add model checkpointing and resuming training

## License

[MIT License](LICENSE)
