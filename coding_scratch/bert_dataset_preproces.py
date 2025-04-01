from warnings import filterwarnings
filterwarnings('ignore') 

import torch
from torch.utils.data import Dataset, DataLoader
from bert_model import bertconfig
from transformers import BertTokenizer
import os
from dotenv import load_dotenv
import random
import multiprocessing

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', token=hf_token )
# special token [MASK] ids ---> 103
mask_ratio = 0.15
random_token_ratio = 0.1
full_mask_ratio = 0.8

cls_idx = 101
sep_idx  =102
mask_idx = 103

num_workers = multiprocessing.cpu_count()
print(f'num worker : {num_workers}')

class BertDataset(Dataset):
    """
    In this class I am aim the create a dataset for training loop fro Bert pre-training

    """
    def __init__(self, texts, mask_ratio=0.15, random_token_ratio=0.1):
        super().__init__()
        self.inputs = []
        self.targets = []
        
        self.tokenizer = tokenizer
        self.max_len = bertconfig['max_len']
        self.stride = bertconfig['max_len']
        self.mask_ratio = mask_ratio
        self.random_token_ratio = random_token_ratio

        self.cls_idx = tokenizer.cls_token_id
        self.sep_idx = tokenizer.sep_token_id
        self.mask_idx = tokenizer.mask_token_id
        self.pad_idx = tokenizer.pad_token_id 

        for text in texts:
            inputs_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length= 512)


            for i in range(0,len(inputs_ids)- self.max_len, self.stride):
                input_chunk = inputs_ids[i:i+self.max_len]
                input_chunk = [self.cls_idx] + input_chunk + [self.sep_idx]
                #truncating
                input_chunk = input_chunk[:self.max_len]
                #pading
                padding_len = self.max_len - len(input_chunk)
                input_chunk += [self.pad_idx] * padding_len
                self.targets.append(input_chunk)


                mask_len = int(len(input_chunk) * self.mask_ratio)
                random_idx = random.sample(range(1,len(input_chunk)-1), mask_len) # cano

                for idx in random_idx:
                    input_chunk[idx] = self.mask_idx
                
                random_token_len = int(mask_len*self.random_token_ratio)
                random_idx = random.sample(range(1, len(input_chunk)-1), random_token_len)

                for idx in random_idx:
                    random_token_id = random.randint(0, tokenizer.vocab_size - 1)
                    input_chunk[idx] = random_token_id
                self.inputs.append(input_chunk)


    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]), torch.tensor(self.targets[index])


# These codes were written to provide training on Kaggle.
if os.path.exists("/kaggle/input/tr-news/data.txt"):
    data_path = "/kaggle/input/tr-news/data.txt"
else:
    data_path = "data.txt"  

with open(data_path, "r", encoding="utf-8") as f:
    data = f.readlines()


train_ratio = .9
train_limit = int(len(data) * train_ratio)

train_data = data[:train_limit]
val_data = data[train_limit:]

print(f'len raw train_data :{len(train_data)}')
print(f'len raw val_data :{len(val_data)}')

train_dataset = BertDataset(train_data)
val_dataset = BertDataset(val_data)

print(f'len train dataset : {len(train_dataset)}')
print(f'len valtrain dataset : {len(val_dataset)}')

train_dataloader = DataLoader(
    dataset= train_dataset,
    batch_size= bertconfig['batch_size'],
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataloader = DataLoader(
    dataset= val_dataset,
    batch_size= bertconfig['batch_size'],
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)
