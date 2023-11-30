import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import numpy as np
# Load the data from the DataFrame
train_df = pd.read_csv("./data/raw/sign_mnist_train.csv")


labels = train_df['label'].values
images = train_df.drop('label', axis=1).values
images = images.reshape(-1, 1, 28, 28).astype(np.float32)  # Reshape and convert to float32




test_df = pd.read_csv("./data/raw/test.csv")

old_test_df = pd.read_csv("./data/raw/old_sign_mnist_test.csv")
test_labels = old_test_df['label'].values
test_images = old_test_df.drop('label', axis=1).values
test_images = test_images.reshape(-1, 1, 28, 28).astype(np.float32)  # Reshape and convert to float32



# Convert to PyTorch tensors
labels_tensor = torch.tensor(labels).long()
images_tensor = torch.tensor(images)

# Splitting the dataset
train_images, val_images, train_labels, val_labels = train_test_split(images_tensor, labels_tensor, test_size=0.3, random_state=42)

# Creating DataLoaders for both sets
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

val_dataset = TensorDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# CNN Model
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=25):
        super(SignLanguageCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 1 * 1, 512)  # Correctly adjusted for the flattened size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3)
        x = self.fc2(x)
        return x

# Initialize the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SignLanguageCNN().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Evaluate Model Function
def evaluate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    with torch.no_grad():  # No need to track gradients
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return avg_loss, accuracy


# Training Loop
num_epochs = 1000  # Define the number of epochs
early_stopper = EarlyStopping(patience=15)
best_val_accuracy = 0.0  # Initialize the best validation accuracy


for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on the validation set
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Update learning rate
    scheduler.step(val_loss)

    # Save the model if it has the best validation accuracy so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        model_save_path = f'./data/weights/{epoch+1}_{val_accuracy}_py_torch_cnn.pkl'
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved the new best model with accuracy: {best_val_accuracy:.4f} at epoch {epoch+1}")

    # Early stopping check
    early_stopper(val_loss)
    if early_stopper.early_stop:
        print("Early stopping")
        break
