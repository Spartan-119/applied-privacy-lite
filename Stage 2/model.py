import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionNeuralNet(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(ConvolutionNeuralNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the network
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    Args:
        net: PyTorch model.
        trainloader: DataLoader for the training set.
        optimizer: PyTorch optimizer.
        epochs: Number of training epochs.
        device: Device to which the model and data should be moved (e.g., 'cuda' or 'cpu').
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test_model(net, testloader, device: str):
    """Validate the network on the entire test set.

    Args:
        net: PyTorch model.
        testloader: DataLoader for the test set.
        device: Device to which the model and data should be moved (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple containing loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    return loss, accuracy