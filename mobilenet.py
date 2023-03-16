import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2
from sklearn.model_selection import train_test_split

device = torch.device('mps')

# Hyperparameters
num_epochs = 30
batch_size = 128
learning_rate = 0.001

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image dataset and split into train, validation, and test sets
dataset = ImageFolder(root='dataset3', transform=transform)
train_val_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
train_set, val_set = train_test_split(train_val_set, test_size=0.2, random_state=42)

# Create dataloaders
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Define model
model = mobilenet_v2(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_ftrs, len(train_set)))
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
best_val_accuracy = 0.0
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training loss every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss {loss.item():.4f}')

    # Evaluate model on validation set
    with torch.no_grad():
        correct = 0
        total = 0
        for data, targets in val_loader:
            # Move data to device
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Make predictions
            scores = model(data)
            _, predictions = torch.max(scores.data, 1)

            # Update metrics
            total += targets.size(0)
            correct += (predictions == targets).sum().item()

        val_accuracy = 100 * correct / total
        print(f'Validation accuracy: {val_accuracy:.2f}%')

        # Save model if validation accuracy is improved
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')

    # Evaluate model on test set
    with torch.no_grad():
        correct = 0
        total = 0
        for data, targets in test_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Make predictions
            scores = model(data)
            _, predictions = torch.max(scores.data, 1)

            # Update metrics
            total += targets.size(0)
            correct += (predictions == targets).sum().item()

        accuracy = 100 * correct / total
        print(f'Test accuracy: {accuracy:.2f}%')
    # Move data to device
