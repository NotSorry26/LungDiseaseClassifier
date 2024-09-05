from matplotlib import pyplot as plt
import numpy as np

# Plots a grid of images from a DataLoader
def plot_data_loader(data_loader, gridDims):
    
    fig, axes = plt.subplots(nrows=gridDims[0], ncols=gridDims[1], figsize=(5, 5))
    dataiter = iter(data_loader)

    for i in range(gridDims[0]):
        for j in range(gridDims[1]):
            try:
                # Fetch the next batch of images
                images, labels = dataiter.next()
                # Convert the image tensor to a numpy array and transpose the dimensions
                image = np.transpose(images[0].numpy(), (1, 2, 0))
                # Display the image on the corresponding subplot
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].axis('off')  # Hide axes ticks
                axes[i, j].set_title(f"Label: {labels[0].item()}") # Add a title to each subplot indicating the label

            except StopIteration:
                # In case the DataLoader runs out of images
                axes[i, j].axis('off')

    plt.tight_layout(pad=1.0)  # Add padding between images
    plt.show()