import torch


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Loads the latest checkpoint from the given path and restores the model and optimizer states.

    Args:
        checkpoint_path (Path): The path to the checkpoint file
        model (nn.Module): A neural network model
        optimizer (torch.optim.Optimizer): The optimizer used for training
    Returns:
        model (nn.Module): The model with restored state
        optimizer (torch.optim.Optimizer): The optimizer with restored state
        epoch (int): The epoch at which the checkpoint was saved
        global_step (int): The global step at which the checkpoint was saved
        tokens_seen (int): The number of tokens seen at the time of saving the checkpoint
        train_losses (list): A list of training loss values for each epoch
        train_ppls (list): A list of training perplexity values for each epoch
        val_losses (list): A list of validation loss values for each epoch
        val_ppls (list): A list of validation perplexity values for each epoch
        all_tokens_seen (list): A list of the number of tokens seen by the model after each epoch
        best_val_loss (float32): The best validation loss seen so far
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    tokens_seen = checkpoint['tokens_seen']
    train_losses = checkpoint['train_losses']
    train_ppls = checkpoint['train_ppls']
    val_losses = checkpoint['val_losses']
    val_ppls = checkpoint['val_ppls']
    all_tokens_seen = checkpoint['all_tokens_seen']
    best_val_loss = checkpoint['best_val_loss']

    return model, optimizer, epoch, global_step, tokens_seen, train_losses, train_ppls, val_losses, val_ppls, all_tokens_seen, best_val_loss

def save_checkpoint(model, optimizer, epoch, global_step, tokens_seen, train_losses, train_ppls, val_losses, val_ppls, all_tokens_seen, checkpoint_path,
                    best_val_loss, name):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'tokens_seen': tokens_seen,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'train_ppls': train_ppls,
        'val_losses': val_losses,
        'val_ppls': val_ppls,
        'all_tokens_seen': all_tokens_seen
    }
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path / f'{name}.pt')
    # also save the best model based on validation loss