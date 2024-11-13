import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load dataset (e.g., MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Build the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 10)       # Second fully connected layer (output layer)
        self.relu = nn.ReLU()               # ReLU activation function

    def forward(self, x):
        x = x.view(-1, 28 * 28)              # Flatten the input image
        x = self.relu(self.fc1(x))           # Apply first layer and ReLU activation
        x = self.fc2(x)                      # Apply second layer
        return x

model = SimpleNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()         # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters()) # Adam optimizer

# Train the model
for epoch in range(5):  # Training for 5 epochs
    for data, target in train_loader:
        optimizer.zero_grad()   # Zero the gradients
        output = model(data)     # Get model output
        loss = criterion(output, target)  # Compute loss
        loss.backward()          # Backpropagation
        optimizer.step()         # Update the weights

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():  # No need to calculate gradients during testing
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)  # Get the predicted class
        total += target.size(0)
        correct += (predicted == target).sum().item()  # Count correct predictions

print(f'Test accuracy: {correct / total}')
