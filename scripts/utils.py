# This script contains utility functions that are used in the main scripts: model & train.

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, trainloader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    print('Training complete')
    return losses

def evaluate_model(model, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

def visualize_augmentations(dataset_original, dataset_augmented, num_images=5):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 10))
    for i in range(num_images):
        index = np.random.randint(len(dataset_original))
        original_img, _ = dataset_original[index]
        augmented_img, _ = dataset_augmented[index]
        
        axes[i, 0].imshow(np.transpose(original_img.numpy(), (1, 2, 0)))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(np.transpose(augmented_img.numpy(), (1, 2, 0)))
        axes[i, 1].set_title("Augmented")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()

def plot_loss_curves(losses_original, losses_augmented):
    plt.figure(figsize=(8,6))
    plt.plot(losses_original, label='Original Data')
    plt.plot(losses_augmented, label='Augmented Data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.show()