import time
import torch
import copy
import numpy as np

def train_model(device, model, dataloaders, optimizer, scheduler, criterion, 
                num_epochs, num_classes, early_stopping_patience, min_delta):
    
    start = time.time()

    # History lists for tracking performance over epochs
    val_accuracy, val_loss = [], []
    val_predictions, val_labels = [], []
    train_accuracy, train_loss = [], []
    train_predictions = []
    train_labels = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    total_steps = len(dataloaders['train'])
    early_stop_count = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss, running_corrects = 0.0, 0
            predictions_batch, labels_batch = [], []

            # Iterate over the data
            for i, (inputs, labels) in enumerate(dataloaders[phase], start=1):
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only in the training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                predictions_batch.append(preds.cpu().detach().numpy())
                labels_batch.append(labels.cpu().detach().numpy())

                if i % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():.4f}, Accuracy: {torch.sum(preds == labels.data).item() / preds.size(0):.2f}%')

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the best model
            if phase == 'val':
                val_accuracy.append(epoch_acc.item())
                val_loss.append(epoch_loss)
                val_predictions.append(np.concatenate(predictions_batch))
                val_labels.append(np.concatenate(labels_batch))

                # Check for improvement and update best model weights
                if epoch_acc > best_acc + min_delta:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_count = 0  # Reset the early stopping counter
                else:
                    early_stop_count += 1  # Increment the early stopping counter

                # Check if early stopping criteria are met
                if early_stop_count >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    time_elapsed = time.time() - start
                    print(f'Training stopped early at {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    model.load_state_dict(best_model_wts)
                    return model, {
                        'acc': val_accuracy, 
                        'loss': val_loss, 
                        'targets': val_labels, 
                        'outputs': val_predictions
                    }, {
                        'acc': train_accuracy, 
                        'loss': train_loss, 
                        'targets': train_labels, 
                        'outputs': train_predictions
                    }
            else:
                train_accuracy.append(epoch_acc.item())
                train_loss.append(epoch_loss)
                train_predictions.append(np.concatenate(predictions_batch))
                train_labels.append(np.concatenate(labels_batch))

        # Step the learning rate scheduler
        scheduler.step()
        print()

    # Calculate total training time
    time_elapsed = time.time() - start
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, {
        'acc': val_accuracy, 
        'loss': val_loss, 
        'targets': val_labels, 
        'outputs': val_predictions
    }, {
        'acc': train_accuracy, 
        'loss': train_loss, 
        'targets': train_labels, 
        'outputs': train_predictions
    }