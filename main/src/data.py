import os
import torchvision.transforms as T
from torchvision import datasets
import torch.utils.data as td
from typing import Dict

# Load training, validation, and test datasets using ImageFolder and DataLoader
def load_data(path: str, batch_size: int, input_size: int, norm_arr: list, num_workers: int = 0) -> Dict[str, td.DataLoader]:

    # Define data transformations for training and validation/test sets
    transform_dict = {
        "train": T.Compose([
            T.Resize(size=input_size),      # Resize images to the specified input size
            T.RandomHorizontalFlip(),       # Randomly flip images horizontally
            T.RandomAdjustSharpness(2),     # Adjust image sharpness
            T.RandomAutocontrast(),         # Automatically enhance the contrast
            T.ToTensor(),                   # Convert images to tensors
            T.Normalize(*norm_arr)          # Normalise images using the provided mean and std
        ]),
        "test_val": T.Compose([
            T.Resize(size=input_size),      # Resize images to the specified input size
            T.ToTensor(),                   # Convert images to tensors
            T.Normalize(*norm_arr)          # Normalise images using the provided mean and std
        ])
    }

    # Load the datasets from the specified directories
    train_dataset = datasets.ImageFolder(root=os.path.join(path, "train"),
                                         transform=transform_dict["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(path, "val"),
                                       transform=transform_dict["test_val"])
    test_dataset = datasets.ImageFolder(root=os.path.join(path, "test"),
                                        transform=transform_dict["test_val"])

    # Create DataLoaders for each dataset
    data_loader_train = td.DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,              # Shuffle the data during training
                                      drop_last=False,           # Do not drop the last batch
                                      num_workers=num_workers,   # Number of subprocesses for data loading
                                      pin_memory=True)           # Pin memory to speed up data transfer to GPU
    data_loader_val = td.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,              # No need to shuffle validation data
                                    drop_last=False,
                                    num_workers=num_workers,
                                    pin_memory=True)
    data_loader_test = td.DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,              # No need to shuffle test data
                                     drop_last=False,
                                     num_workers=num_workers,
                                     pin_memory=True)

    # Return a dictionary with DataLoaders for train, val, and test sets
    return {
        'train': data_loader_train,
        'val': data_loader_val,
        'test': data_loader_test
    }