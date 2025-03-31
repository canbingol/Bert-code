import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from bert_model import BERT, bertconfig
from bert_dataset_preproces import train_dataloader, val_dataloader

# Cihaz kontrolü
device = torch.device(bertconfig['device'] if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

# Model oluşturma
model = BERT(bertconfig).to(device)
print(f"Model yüklendi ve {device} cihazına taşındı")

# Toplam parametre sayısını hesaplama
total_params = sum(p.numel() for p in model.parameters())
print(f"Toplam parametre sayısı: {total_params:,}")

# Hiperparametreler
EPOCHS = 5
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

# Optimizer ve loss fonksiyonu
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad_idx'i yoksay (0 olarak varsayıyorum)

# Learning rate scheduler
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Toplam adım sayısını hesapla
total_steps = len(train_dataloader) * EPOCHS

# Scheduler oluştur
scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)

# Model kaydetme yolu
MODEL_SAVE_PATH = 'bert_checkpoints'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Eğitim ve değerlendirme sırasında metrikleri takip etmek için
train_losses = []
val_losses = []
train_perplexities = []
val_perplexities = []

# Eğitim fonksiyonu
def train(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    epoch_loss = 0
    correct_preds = 0
    total_preds = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Çıktı boyutunu yeniden düzenle: [batch_size, seq_len, vocab_size] -> [batch_size*seq_len, vocab_size]
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        # Loss hesapla
        loss = criterion(outputs, targets)
        
        # Backward pass ve optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (aşırı gradyanları sınırla)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Metrikleri güncelle
        epoch_loss += loss.item()
        
        # Perplexity için doğru tahminleri hesapla (sadece pad olmayan tokenlarda)
        mask = targets != 0  # pad idx'i hariç tut
        _, predicted = torch.max(outputs[mask], 1)
        total_preds += mask.sum().item()
        correct_preds += (predicted == targets[mask]).sum().item()
        
        # Progress bar'ı güncelle
        progress_bar.set_postfix({
            'loss': f'{epoch_loss/(batch_idx+1):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, accuracy, perplexity

# Değerlendirme fonksiyonu
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Çıktı boyutunu yeniden düzenle
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            # Loss hesapla
            loss = criterion(outputs, targets)
            
            # Metrikleri güncelle
            epoch_loss += loss.item()
            
            # Perplexity için doğru tahminleri hesapla (sadece pad olmayan tokenlarda)
            mask = targets != 0  # pad idx'i hariç tut
            _, predicted = torch.max(outputs[mask], 1)
            total_preds += mask.sum().item()
            correct_preds += (predicted == targets[mask]).sum().item()
            
            # Progress bar'ı güncelle
            progress_bar.set_postfix({'loss': f'{epoch_loss/(batch_idx+1):.4f}'})
    
    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, accuracy, perplexity

# Eğitim döngüsü
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # Eğitim
    train_loss, train_acc, train_ppl = train(model, train_dataloader, optimizer, criterion, scheduler, device)
    train_losses.append(train_loss)
    train_perplexities.append(train_ppl)
    
    # Değerlendirme
    val_loss, val_acc, val_ppl = evaluate(model, val_dataloader, criterion, device)
    val_losses.append(val_loss)
    val_perplexities.append(val_ppl)
    
    # Epoch süresini hesapla
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    # Sonuçları yazdır
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc*100:.2f}% | Train PPL: {train_ppl:.2f}')
    print(f'\tVal. Loss: {val_loss:.3f} | Val. Accuracy: {val_acc*100:.2f}% | Val. PPL: {val_ppl:.2f}')
    
    # En iyi modeli kaydet
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_perplexity': train_ppl,
            'val_perplexity': val_ppl,
            'config': bertconfig
        }, os.path.join(MODEL_SAVE_PATH, 'best_model.pt'))
        print(f"En iyi model kaydedildi (Epoch {epoch+1}, Val. Loss: {val_loss:.3f})")
    
    # Her epoch sonunda checkpoint kaydet
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_perplexity': train_ppl,
        'val_perplexity': val_ppl,
        'config': bertconfig
    }, os.path.join(MODEL_SAVE_PATH, f'checkpoint_epoch_{epoch+1}.pt'))

# Eğitim sonuçlarını görselleştir
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(train_perplexities, label='Train Perplexity')
plt.plot(val_perplexities, label='Validation Perplexity')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.legend()
plt.title('Training and Validation Perplexity')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_results.png'))
plt.show()

print(f"Eğitim tamamlandı! Checkpoint dosyaları '{MODEL_SAVE_PATH}' dizinine kaydedildi.")

# Eğitim süresi bilgisi
print(f"Toplam eğitim süresi: {datetime.timedelta(seconds=int(time.time() - start_time))}")

# Devam etmek veya tekrar eğitmek için bir fonksiyon
def resume_training(checkpoint_path, train_dataloader, val_dataloader, epochs=5):
    """
    Eğitimi bir checkpoint'ten devam ettirmek için kullanılabilir
    """
    # Checkpoint'i yükle
    checkpoint = torch.load(checkpoint_path)
    
    # Modeli oluştur ve ağırlıkları yükle
    model = BERT(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizer'ı yeniden oluştur ve durumunu yükle
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Schedulerı yeniden oluştur ve durumunu yükle
    total_steps = len(train_dataloader) * epochs
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Eğitim {checkpoint_path} dosyasından devam ediyor (Epoch {checkpoint['epoch']})")
    
    # Eğitimi başlat
    for epoch in range(checkpoint['epoch'], checkpoint['epoch'] + epochs):
        # (Eğitim kodunu burada tekrarla)
        pass
