import torch
import time
from src.evaluate import evaluate_model

def train_model(model, trainloader, testloader, criterion, optimizer, scheduler=None, num_epochs=10, device='cpu'):
    """
    Trains the CNN model and returns history.
    """
    model.to(device)
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        epoch_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        if scheduler:
            scheduler.step()
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        avg_loss = epoch_loss / len(trainloader)
        history['train_loss'].append(avg_loss)
        
        # Evaluate after each epoch
        acc, _, _ = evaluate_model(model, testloader, device=device)
        history['val_acc'].append(acc)

        
        print(f'Epoch {epoch+1} completed in {time.time() - start_time:.2f}s | Avg Loss: {avg_loss:.3f} | Val Acc: {acc:.2f}%')

    print('Finished Training')
    torch.save(model.state_dict(), 'cifar_net.pth')
    print('Model saved to cifar_net.pth')
    return history



