def train(
    model, device, dataloader, 
    n_batches: int, epoch: int,
    loss_funcs: dict, optimizer, clip=10.,
    last_n_losses=500, verbose=True):

    losses_midas = list() # calc for midas only
    losses_entity = list() # calc for entity only
    losses_sum = list() # sum up midas and entity losses to get one loss
    losses_concat = list() # calc for midas_entity combination 

    progress_bar = tqdm(total=n_batches, disable=not verbose, desc=f'Train epoch {epoch}')
    
    model.train()

    for x, y_multi, y_concat in dataloader:

        optimizer.zero_grad()
        
        x = x.to(device)
        
        y_midas = y_multi[:,0].to(device)
        y_entity = y_multi[:,1].to(device)
        y_concat = y_concat.to(device)
        
        logits_midas, logits_entity, logits_concat = model(x)
        
        loss_midas = loss_funcs['midas'](logits_midas, y_midas)
        loss_entity = loss_funcs['entity'](logits_entity, y_entity)
        loss_concat = loss_funcs['concat'](logits_concat, y_concat)
        
        print(loss_midas, loss_entity, loss_concat)
        
        # Perform backpropatation
        loss_midas.backward()
        loss_entity.backward()
        loss_concat.backward()
        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        losses_midas.append(loss_midas.item())
        losses_entity.append(loss_entity.item())
        losses_sum.append(loss_midas.item() + loss_entity.item())
        losses_concat.append(loss_concat.item())
        
        progress_bar.set_postfix(
            loss_midas=np.mean(losses_midas[-last_n_losses:]),
            loss_entity=np.mean(losses_entity[-last_n_losses:]),
            loss_concat=np.mean(losses_concat[-last_n_losses:])
        )

        progress_bar.update()

    progress_bar.close()
    
    loss_epoch = [
        np.mean(losses_midas), np.mean(losses_entity), 
        np.mean(losses_concat), np.mean(losses_sum)
    ]
    
    return loss_epoch


def evaluate(
    model, device, dataloader, 
    n_batches: int, epoch: int, loss_funcs: dict, last_n_losses=500, verbose=True):

    losses_midas = list() # calc for midas only
    losses_entity = list() # calc for entity only
    losses_sum = list() # sum up midas and entity losses to get one loss
    losses_concat = list() # calc for midas_entity combination
    
    preds_midas, preds_entity, preds_concat = list(), list(), list()
    true_midas, true_entity, true_concat = list(), list(), list()
    
    progress_bar = tqdm(total=n_batches, disable=not verbose, desc=f'Val epoch {epoch}')

    model.eval()
    
    for x, y_multi, y_concat in valid_loader:
        
        x = x.to(device)
        y_midas = y_multi[:,0].to(device)
        y_entity = y_multi[:,1].to(device)
        y_concat = y_concat.to(device)
        
        with torch.no_grad():
            logits_midas, logits_entity, logits_concat = model(x)
        
        # get losses
        loss_midas = loss_funcs['midas'](pred_midas, y_midas)
        loss_entity = loss_funcs['entity'](pred_entity, y_entity)
        loss_concat = loss_funcs['concat'](pred_concat, y_concat)
        
        losses_midas.append(loss_midas.item())
        losses_entity.append(loss_entity.item())
        losses_sum.append(loss_midas.item() + loss_entity.item())
        losses_concat.append(loss_concat.item())
        
        pred_midas_batch = torch.argmax(torch.softmax(logits_midas, -1), -1).cpu().tolist()
        pred_entity_batch = torch.argmax(torch.softmax(logits_entity, -1), -1).cpu().tolist()
        pred_concat_batch = torch.argmax(torch.softmax(logits_concat, -1), -1).cpu().tolist()
        
        true_midas_batch = y_midas.cpu().tolist()
        true_entity_batch = y_entity.cpu().tolist()
        true_concat_batch = y_concat.cpu().tolist()
        
        # add up to the total list of predictions
        preds_midas += pred_midas_batch
        preds_entity += pred_entity_batch
        preds_concat += pred_concat_batch
        
        progress_bar.set_postfix(
            loss_midas=np.mean(losses_midas[-last_n_losses:]),
            loss_entity=np.mean(losses_entity[-last_n_losses:]),
            loss_concat=np.mean(losses_concat[-last_n_losses:])
        )
        progress_bar.update()
    
    progress_bar.close()

    f1_midas = f1_score(true_midas, preds_midas, average='weighted')
    acc_midas = accuracy_score(true_midas, preds_midas)
    
    f1_entity = f1_score(true_entity, preds_entity, average='weighted')
    acc_entity = accuracy_score(true_entity, preds_entity)
    
    f1_concat = f1_score(true_concat, preds_concat, average='weighted')
    acc_concat = accuracy_score(true_concat, preds_concat)
    
    f1_av = np.mean([f1_ep_midas, f1_ep_entity])
    acc_av = np.mean([acc_ep_midas, acc_ep_entity])
    
    f1_epoch = [f1_midas, f1_entity, f1_concat, f1_av]
    acc_epoch = [acc_midas, acc_entity, acc_concat, acc_av]
    
    loss_epoch = [
        np.mean(losses_midas), np.mean(losses_entity), 
        np.mean(losses_concat), np.mean(losses_sum)
    ]
    
    return loss_epoch, f1_epoch, acc_epoch