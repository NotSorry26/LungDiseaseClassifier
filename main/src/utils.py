import os
import torch
from typing import List, Union

# Set the `requires_grad` attribute of the model's parameters
def set_requires_grad(model: torch.nn.Module, feature_extracting: bool) -> None:

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Save the model to a specified path            
def save_model(model: torch.nn.Module, path: str, name: str) -> bool:

    torch.save(model, os.path.join(path, name))
    return True

# Encode a list of class labels into a tensor with one-hot encoding
def encode_label(label: List[str], classes_list: List[str]) -> torch.Tensor:

    target = torch.zeros(len(classes_list))  # Initialize tensor with zeros
    for l in label:
        idx = classes_list.index(l)  # Find index of label in classes list
        target[idx] = 1  # Set corresponding position to 1
    return target

# Decode a prediction tensor into human-readable labels
def decode_target(classes: List[str], target: Union[torch.Tensor, List[float]], threshold: float = 0.5) -> str:

    result = [classes[i] for i, x in enumerate(target) if x >= threshold]  # Collect class names where target exceeds threshold
    return ' '.join(result)  # Join class names into a single string