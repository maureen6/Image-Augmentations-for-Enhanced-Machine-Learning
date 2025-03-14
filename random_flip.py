import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Get a single image from the dataset
image, label = dataset[0]

# Define the RandomFlip transformations
horizontal_flip = transforms.RandomHorizontalFlip(p=1)  # Always flip horizontally
vertical_flip = transforms.RandomVerticalFlip(p=1)  # Always flip vertically

# Apply transformations
horizontally_flipped = horizontal_flip(image)
vertically_flipped = vertical_flip(image)

# Plot original and flipped images
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(image.permute(1, 2, 0))
axes[0].set_title("Original")

axes[1].imshow(horizontally_flipped.permute(1, 2, 0))
axes[1].set_title("Horizontal Flip")

axes[2].imshow(vertically_flipped.permute(1, 2, 0))
axes[2].set_title("Vertical Flip")

for ax in axes:
    ax.axis("off")

plt.show()
