import torch  # PyTorch
import numpy as np  # For handling arrays

# Metrics and Plots
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For creating heatmaps

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# Evaluate the model on the test dataset
def eval_model(device: torch.device, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> dict:

    model.eval()

    running_corrects = 0
    outputs_batch = []
    targets_batch = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            _, preds = torch.max(probabilities, 1)

            running_corrects += torch.sum(preds == labels.data)
            outputs_batch.append(probabilities.cpu().detach().numpy())
            targets_batch.append(labels.cpu().detach().numpy())

    outputs_batch = np.concatenate(outputs_batch)
    targets_batch = np.concatenate(targets_batch)

    test_acc = running_corrects.double() / len(test_loader.dataset)
    test_f1 = f1_score(targets_batch, np.argmax(outputs_batch, axis=1), average='macro')
    test_precision = precision_score(targets_batch, np.argmax(outputs_batch, axis=1), average='macro')
    test_recall = recall_score(targets_batch, np.argmax(outputs_batch, axis=1), average='macro')
    test_cm = confusion_matrix(targets_batch, np.argmax(outputs_batch, axis=1))

    # ROC AUC for multiclass classification using one-vs-rest
    roc_auc = roc_auc_score(targets_batch, outputs_batch, multi_class="ovr", average="macro")

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return {
        'acc': test_acc.item(),
        'f1': test_f1,
        'precision': test_precision,
        'recall': test_recall,
        'cm': test_cm,
        'roc_auc': roc_auc,
        'outputs': np.argmax(outputs_batch, axis=1),
        'targets': targets_batch
    }