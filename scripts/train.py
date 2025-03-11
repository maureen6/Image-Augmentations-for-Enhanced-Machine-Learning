import torch
import torchvision
import torchvision.transforms as transforms

from model import CNN
from utils import train_model, evaluate_model, visualize_augmentations, plot_loss_curves


# TRANSFORMATIONS  & IMAGE AUGMENTATION TECHNIQUES TO PERFORM ON THE DATA

transform_original = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_augmented = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=(3,3)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# LOADING ORIGINAL CIFAR-10 DATASET AND PERFORMING AUGMENTATION

batch_size = 64

trainset_original = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_original)
trainloader_original = torch.utils.data.DataLoader(trainset_original, batch_size=batch_size, shuffle=True)

trainset_augmented = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)
trainloader_augmented = torch.utils.data.DataLoader(trainset_augmented, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_original)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# VISUALIZING THE ORIGINAL IMAGES VS THE AUGMENTED IMAGES
visualize_augmentations(trainset_original, trainset_augmented)

# TRAINING THE MODEL ON ORIGINAL AND AUGMENTED DATASET
print("Training on original dataset...")
model_original = CNN()
losses_original = train_model(model_original, trainloader_original)

print("Training on augmented dataset...")
model_augmented = CNN()
losses_augmented = train_model(model_augmented, trainloader_augmented)

# EVALUATING THE MODEL ON TEST DATA FROM BOTH DATASETS
evaluate_model(model_original, testloader)
evaluate_model(model_augmented, testloader)

# PLOTTING THE LOSS CURVES FOR BOTH MODELS
plot_loss_curves(losses_original, losses_augmented)

