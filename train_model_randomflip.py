import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*4*4, 512)  # After pooling, the size becomes 4x4 (CIFAR-10)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)

        # Flatten dynamically
        x = x.view(x.size(0), -1)  # Flatten the output dynamically
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data Preprocessing and Augmentation
transform_no_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 Dataset
trainset_no_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_no_aug)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_no_aug)

trainloader_no_aug = DataLoader(trainset_no_aug, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

trainset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_aug)
trainloader_aug = DataLoader(trainset_aug, batch_size=64, shuffle=True)

# Training Function
def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=5):
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_losses.append(running_loss / len(trainloader))
        test_accuracy = evaluate_model(model, testloader)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return train_losses, test_accuracies

# Evaluation Function
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# Initialize Models, Loss, Optimizer
model_no_aug = SimpleCNN()
model_aug = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer_no_aug = optim.SGD(model_no_aug.parameters(), lr=0.001, momentum=0.9)
optimizer_aug = optim.SGD(model_aug.parameters(), lr=0.001, momentum=0.9)

# Train Models
print("Training on original dataset without augmentation...")
train_losses_no_aug, test_accuracies_no_aug = train_model(model_no_aug, trainloader_no_aug, testloader, criterion, optimizer_no_aug)

print("\nTraining on augmented dataset...")
train_losses_aug, test_accuracies_aug = train_model(model_aug, trainloader_aug, testloader, criterion, optimizer_aug)

# Compare Results
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_no_aug, label='No Augmentation')
plt.plot(train_losses_aug, label='With Augmentation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies_no_aug, label='No Augmentation')
plt.plot(test_accuracies_aug, label='With Augmentation')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy (%)')
plt.legend()

plt.show()

# Save the model if necessary
torch.save(model_no_aug.state_dict(), 'model_no_aug.pth')
torch.save(model_aug.state_dict(), 'model_aug.pth')
