import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score 
from torch import nn
from tqdm.notebook import tqdm

class BaseModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, n_classes: int, batch_size: int = 32):
        super().__init__()
        
        # parameters
        self.input_size = input_size
        self.output_size = n_classes
        self.batch_size = batch_size

        # layers
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        # classifiers
        self.clf = nn.Linear(hidden_size, n_classes)
        

    def forward(self, x: torch.Tensor):
        x = self.linear_in(x)
        x = self.relu(x)
        pred = self.clf(x)
        return pred
    
    
def train_single_model(
    model, device, dataloader, 
    n_batches: int, epoch: int,
    loss_fn: dict, optimizer, clip=10.,
    last_n_losses=500, verbose=True):

    losses = list() 

    progress_bar = tqdm(total=n_batches, disable=not verbose, desc=f'Train epoch {epoch}')
    
    model.train()

    for x, _, y_concat in dataloader:

        optimizer.zero_grad()

        x = x.to(device)
        y_concat = y_concat.to(device)
        
        logits = model(x)
        
        loss = loss_fn(logits, y_concat)
        
        # backpropatation
        loss.backward()
        
        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()

        losses.append(loss.item())
        
        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]))

        progress_bar.update()

    progress_bar.close()
    
    return losses


def evaluate_single_model(
    model, device, dataloader, n_batches: int, epoch: int, loss_fn: dict, last_n_losses=500, verbose=True):

    losses = list()
    
    preds = list()
    y_true = list()
    
    progress_bar = tqdm(total=n_batches, disable=not verbose, desc=f'Val epoch {epoch}')

    model.eval()
    
    for x, _, y_concat in dataloader:
        
        x = x.to(device)
        y_concat = y_concat.to(device)
        
        with torch.no_grad():
            logits = model(x)
        
        # get losses
        loss = loss_fn(logits, y_concat)
        losses.append(loss.item())
        
        y_pred_batch = torch.argmax(torch.softmax(logits, -1), -1).cpu().tolist()
        y_true_batch = y_concat.cpu().tolist()
        
        # add up to the total list of predictions and targets
        preds += y_pred_batch
        y_true += y_true_batch
        
        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]))
        progress_bar.update()
    
    progress_bar.close()

    f1_epoch = f1_score(y_true, preds, average='weighted')
    acc_epoch = accuracy_score(y_true, preds)
    
    return losses, f1_epoch, acc_epoch