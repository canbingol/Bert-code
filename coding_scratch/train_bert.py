import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from bert_model import BERT, bertconfig
from bert_dataset_preproces import dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bertconfig['device'] = device
model = BERT(bertconfig)
model = model.to(device)

def calc_loss_batch(input_batch,target_batch):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float('nan')
    elif num_batches is  None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i ,(input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch
            )
            total_loss = loss.item()

        else:
            break
    return total_loss / num_batches


with torch.no_grad():
    train_loss = calc_loss_loader()
    val_loss = calc_loss_loader()

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

train_losses, track_tokens_seen = [], []
tokens_seen, global_step = 0, -1

def bert_train(optimizer, eval_freq: int = 2, epochs: int = 10):
    global tokens_seen, track_tokens_seen, global_step, train_losses
    print(f'dataloader size: {len(dataloader)}')
    #Checkpoint folder
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for input_batch, target_batch in dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch)

            loss.backward()
            optimizer.step()
            
            tokens_seen += input_batch.numel()  
            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    loss = calc_loss_loader()  
                model.train()
                train_losses.append(loss)  
                track_tokens_seen.append(tokens_seen)

                print(f"Epoch [{epoch+1}/{epochs}], Step [{global_step}], Training Loss: {loss:.4f}")
        checkpoint_path = os.path.join(checkpoint_dir, f"bert_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'tokens_seen': tokens_seen,
            'global_step': global_step
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
epochs = 4
eval_freq = 20
bert_train(optimizer, eval_freq, epochs=epochs)



def plot_losses(epochs_seen, tokens_seen, train_losses, file_name=None):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()


    plt.show()


epochs_tensor = torch.linspace(0, epochs - 1, len(train_losses))
plot_losses(epochs_tensor, track_tokens_seen, train_losses, file_name="training_loss_plot.png")
