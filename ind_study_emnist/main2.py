import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Corrected data transforms
transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)  # Invert: black background, white letter
])

# Load EMNIST Letters dataset
train_data = datasets.EMNIST(
    root='.', 
    split='letters',
    train=True,
    download=True,
    transform=transform
)
test_data = datasets.EMNIST(
    root='.', 
    split='letters',
    train=False,
    download=True,
    transform=transform
)

# Adjust labels: map 1-26 to 0-25
train_data.targets = train_data.targets - 1
test_data.targets = test_data.targets - 1

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# Corrected model (26 classes)
class LetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)  # Fixed dimensions
        self.fc2 = nn.Linear(128, 26)  # 26 classes A-Z

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 28x28 -> 28x28
        x = F.max_pool2d(x, 2)     # 14x14
        x = F.relu(self.conv2(x))  # 14x14 -> 14x14
        x = F.max_pool2d(x, 2)     # 7x7
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = LetterCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training function
def train_model(epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), "emnist_cnn_corrected.pth")
    print("Model saved as emnist_cnn_corrected.pth")

# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Quick training (5 minutes on GPU, ~15 on CPU)
train_model(epochs=5)
accuracy = evaluate_model()

# Sample prediction visualization
def show_sample_prediction():
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Display first 5 images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img = images[i].cpu().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {chr(labels[i].item()+65)}\nPred: {chr(preds[i].item()+65)}")
        axes[i].axis('off')
    plt.savefig("sample_predictions.png")
    print("Sample predictions saved as sample_predictions.png")

show_sample_prediction()

print(f"\nTraining complete! Final accuracy: {accuracy:.2f}%")