import torch

from src.utils import generate_text_tokIDs, text_to_tokenIDs, text_to_tokenIDs, tokenIDs_to_text


def calc_loss_batch(model, input_batch, target_batch, device, criterion):
    """
    Calculates the loss and the perplexity for a single batch of input and target data.

    Args:
        model (nn.Module): A neural network model
        input_batch : an input batch from the dataloader
        target_batch : the target batch from the dataloader
        device : the device that the model is currently on (CPU or GPU)
        criterion : the cost function used to calculate the loss

    Returns:
        loss (Pytorch tensor) : the loss value for that batch
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = criterion(logits.flatten(0, 1), target_batch.flatten())  # we flatten the logits (0, 1) to combine batch and sequence dims

    return loss

# num_batches gives the freedom to calculate loss and perplexity across a subset of the dataset
# instead of the whole dataset
# since the dataset is large and calculating loss across the whole dataset can take a long time
def calc_loss_loader(model, data_loader, device, criterion, num_batches):
    """
    Calculates the average loss and perplexity for a given data loader.
    
    Args:
        model (nn.Module): A neural network model
        data_loader (DataLoader): A PyTorch DataLoader object containing the dataset
        device : the device that the model is currently on (CPU or GPU)
        criterion : the cost function used to calculate the loss
        num_batches (int): The number of batches to evaluate. If -1 or None, evaluates over the entire dataset.

    Returns:
        avg_loss (float32) : the average loss value across the evaluated batches
        perplexity (float32) : perplexity value across the evaluated batches
    """
    total_loss, total_perplexity = 0.0, 0.0

    # handle the case at which the data_loader has no data smh
    if len(data_loader) == 0 or num_batches == 0:
        return float('nan'), float('nan')

    if num_batches is None or num_batches == -1:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))        # min between number of batches and length dataset (to avoid num batches being > len dataset)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(model, input_batch, target_batch, device, criterion)
        total_loss += loss.item() 
        avg_loss = total_loss / num_batches

        perplexity += torch.exp(avg_loss).item()
    
    return avg_loss, perplexity  # return average loss and perplexity across the batches

# we can now define a function that evaluates the model on both the training and validation datasets
def evaluate_model(model, train_loader, val_loader, device, criterion, num_batches):
    model.eval()
    with torch.no_grad():
        train_loss, train_perplexity = calc_loss_loader(model, train_loader, device, criterion, num_batches)
        val_loss, val_perplexity = calc_loss_loader(model, val_loader, device, criterion, num_batches)

    model.train()   
    return train_loss, train_perplexity, val_loss, val_perplexity

def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens, context_length):
    tokenIDs = text_to_tokenIDs(start_context, tokenizer)
    tokenIDs = tokenIDs.to(device)

    model.eval()
    generated_tokIDs = generate_text_tokIDs(model, tokenIDs, max_new_tokens, context_length, tokenizer)
    generated_text = tokenIDs_to_text(generated_tokIDs, tokenizer)
    print(f"Generated text: {generated_text}")
    model.train()