from bert_dataset_preproces import train_dataloader, val_dataloader
from bert_model import BERT, bertconfig

import torch.nn as nn
import torch.optim as optim
import tqdm
import torch
import time
import os
import matplotlib.pyplot as plt
import datetime

EPOCHS = 2
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MODEL_SAVE_PATH = 'bert_checkpoints'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

device = bertconfig['device']
model = BERT(bertconfig)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model params: {total_params:,}")

criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore pad_idx
optimizer = optim.AdamW(model.parameters(), lr=LR, eps=2e-5, weight_decay=WEIGHT_DECAY)


train_losses, val_losses = [],[]
def train_model():
    model.train()

    epoch_loss = 0

    progress_bar = tqdm.tqdm(train_dataloader,desc='Training')

    for batch_idx ,(train_batch, target_batch) in enumerate(progress_bar):
        train_batch = train_batch.to(device)
        target_batch = target_batch.to(device)

        #forward pass
        output = model(train_batch)

        output = output.view(-1, output.size(-1))
        target_batch = target_batch.view(-1)

        loss = criterion(output, target_batch)

        # backwrad pass
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        progress_bar.set_postfix({
            'loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })
        
    avg_loss = epoch_loss / len(train_dataloader)

    return avg_loss


def evaulate_model():
    model.eval()

    epoch_loss = 0

    progress_bar = tqdm.tqdm(val_dataloader,desc='Validation')

    with torch.no_grad():

        for batch_idx ,(train_batch, target_batch) in enumerate(progress_bar):
            train_batch = train_batch.to(device)
            target_batch = target_batch.to(device)

            #forward pass
            output = model(train_batch)

            output = output.view(-1, output.size(-1))
            target_batch = target_batch.view(-1)

            loss = criterion(output, target_batch)

            epoch_loss += loss.item()

            progress_bar.set_postfix({
                'loss': f'{epoch_loss/(batch_idx+1):.4f}'
                })
        
    avg_loss = epoch_loss / len(train_dataloader)
    model.train()
    return avg_loss


for epoch in range(EPOCHS):
    best_val_loss = float('inf')

    start_time = time.time()

    train_loss = train_model()
    train_losses.append(train_loss)

    val_loss = evaulate_model()
    val_losses.append(val_loss)

    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)


    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.3f} ')
    print(f'\tVal. Loss: {val_loss:.3f} ')


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': bertconfig
        }, os.path.join(MODEL_SAVE_PATH, 'best_model.pt'))
        print(f"Best model saved (Epoch {epoch+1}, Val. Loss: {val_loss:.3f})")



plt.figure(figsize=(12, 5))

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')


plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_results.png'))
plt.show()

print(f"training finished! Checkpoint folder '{MODEL_SAVE_PATH}' .")

# Eğitim süresi bilgisi
print(f"Total training time: {datetime.timedelta(seconds=int(time.time() - start_time))}")
