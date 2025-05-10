# BERT Implementation and Fine-tuning Projects

This repository contains multiple BERT (Bidirectional Encoder Representations from Transformers) implementations and applications:

1. **BERT from Scratch**: A PyTorch implementation of BERT built from the ground up
2. **BERT Fine-tuning with Adapter Layers**: A project demonstrating how to fine-tune pre-trained BERT models using adapter layers
3. **coding_scratch**: Directory containing the complete implementation of BERT from scratch with detailed documentation

## BERT Fine-tuning with Adapter Layers

This project demonstrates how to fine-tune a pre-trained BERT model using adapter layers for sarcasm detection in news headlines.

### Project Overview

- Uses pre-trained BERT model from Hugging Face
- Implements adapter layers to efficiently fine-tune the model
- Applied to sarcasm detection in news headlines
- Notebook demonstrates the entire workflow from data preprocessing to evaluation

### Dataset

The project uses the News Headlines Dataset for Sarcasm Detection, which contains:
- News headlines labeled as sarcastic or non-sarcastic
- Over 26,000 headlines from The Onion and HuffPost

### Implementation Details

- Freezes the main BERT model parameters
- Adds custom adapter layers for fine-tuning
- Trains the model on the sarcasm detection task
- Evaluates performance on validation and test sets

### Results

The fine-tuned model achieves approximately 85% training accuracy and 44% test accuracy in sarcasm detection, demonstrating the effectiveness of adapter-based fine-tuning for this task.

## BERT from Scratch

For details on the from-scratch BERT implementation including model architecture, pre-training approach, and usage instructions, please see:
[BERT from Scratch README](https://github.com/canbingol/Bert-code/blob/master/coding_scratch/README.md)

The from-scratch implementation includes:
- Complete BERT architecture implementation in PyTorch
- Masked Language Modeling pre-training
- Custom dataset preprocessing for training

## coding_scratch Directory

The `coding_scratch` directory contains the full codebase for the BERT from scratch implementation. This includes:

- Model architecture definition
- Tokenization and data preprocessing
- Training scripts
- Utility functions

For detailed documentation on the BERT from scratch implementation, see:
[coding_scratch README](https://github.com/canbingol/Bert-from-scratch/blob/main/README.md)
Then, run the training script:
```
python train_bert.py
```
##  How to Use

Clone the repository and navigate to the working directory:

```bash
git clone https://github.com/canbingol/Bert-code.git
cd Bert-code/coding_scratch
```
## Requirements

- PyTorch
- Transformers (for Hugging Face models)
- NumPy
- Matplotlib

## Usage

Each project has its own specific usage instructions. See the individual notebooks and READMEs for details.

## License

[MIT License](LICENSE)
