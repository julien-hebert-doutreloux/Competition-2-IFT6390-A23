import pandas as pd
import numpy as np
import torch.nn.functional as F
# Importing the required packages
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import f1_score
import seaborn as sns
import os 
train_df = pd.read_csv("./data/raw/sign_mnist_train.csv").sample(n=100)
old_test_df = pd.read_csv("./data/raw/old_sign_mnist_test.csv")
test_df = pd.read_csv("./data/raw/test.csv")

class GestureDataset(Dataset):
    def __init__(self,csv,train=True):
        self.csv=pd.read_csv(csv).sample(frac=1)
        self.csv.reset_index(drop=True, inplace=True)
        self.img_size=224
        # print(self.csv['image_names'][:5])
        self.train=train
        text="pixel"
        self.images=torch.zeros((self.csv.shape[0],1))
        for i in range(1,785):
            temp_text=text+str(i)
            temp = self.csv[temp_text].values
            temp=torch.FloatTensor(temp).unsqueeze(1)
            self.images=torch.cat((self.images,temp),1)
        self.labels=self.csv['label']
        self.images=self.images[:,1:]
        self.images=self.images.view(-1,28,28)
        
    def __getitem__(self,index):
        img=self.images[index]
        img=img.numpy()
        img=cv2.resize(img,(self.img_size,self.img_size))
        tensor_image=torch.FloatTensor(img)
        tensor_image=tensor_image.unsqueeze(0)
        tensor_image/=255.
        if self.train:
            return tensor_image,self.labels[index]
        else:
            return tensor_image
    def __len__(self):
        return self.images.shape[0]


# Using custom GestureDataset class to load train and test data respectively.
data=GestureDataset("./data/raw/sign_mnist_train.csv")
data_val=GestureDataset("./data/raw/old_sign_mnist_test.csv")

# Using the in-built DataLoader to create batches of images and labels for training validation respectively. 
train_loader=torch.utils.data.DataLoader(dataset=data,batch_size=128,num_workers=4,shuffle=True)
val_loader=torch.utils.data.DataLoader(dataset=data_val,batch_size=64,num_workers=0,shuffle=True)

class Classifier(nn.Module):
    def __init__(self, num_classes=25):
        super(Classifier, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(1, 32, 5), # 220, 220
        nn.MaxPool2d(2), # 110, 110
        nn.ReLU(),
        nn.BatchNorm2d(32)
        )
        self.Conv2 = nn.Sequential(
        nn.Conv2d(32, 64, 5), # 106, 106
        nn.MaxPool2d(2),  # 53,53
        nn.ReLU(),
        nn.BatchNorm2d(64)
        )
        self.Conv3 = nn.Sequential(
        nn.Conv2d(64, 128, 3), # 51, 51
        nn.MaxPool2d(2), # 25, 25
        nn.ReLU(),
        nn.BatchNorm2d(128)
        )
        self.Conv4 = nn.Sequential(
        nn.Conv2d(128, 256, 3), # 23, 23
        nn.MaxPool2d(2), # 11, 11
        nn.ReLU(),            
        nn.BatchNorm2d(256)
        )
        self.Conv5 = nn.Sequential(
        nn.Conv2d(256, 512, 3), # 9, 9
        nn.MaxPool2d(2), # 4, 4
        nn.ReLU(),
        nn.BatchNorm2d(512)
        )
        
        self.Linear1 = nn.Linear(512 * 4 * 4, 256)
        self.dropout=nn.Dropout(0.1)
        self.Linear3 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x=self.dropout(x)
        x = self.Conv5(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.dropout(x)
        x = self.Linear3(x)
        return x
        
        
# Validating the model against the validation dataset and generate the accuracy and F1-Score.
def validate(val_loader, model, criterion, device):
    model.eval()
    test_labels = [0]
    test_pred = [0]
    losses = []

    for i, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            predicted = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(predicted, 1)
            test_pred.extend(list(predicted.cpu().numpy()))
            test_labels.extend(list(labels.cpu().numpy()))

    test_pred = np.array(test_pred[1:])
    test_labels = np.array(test_labels[1:])
    correct = (test_pred == test_labels).sum()
    accuracy = correct / len(test_labels)
    f1_test = f1_score(test_labels, test_pred, average='weighted')
    average_loss = np.mean(losses)
    
    model.train()
    return accuracy, f1_test, average_loss
    

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

# Define your model, optimizer, criterion, and other necessary components

# Initialize EarlyStopping object
early_stopper = EarlyStopping(patience=3)
model=Classifier()

model.train()
checkpoint=None
device="cpu"
learning_rate=1e-4
start_epoch=0
end_epoch=20
best_val_accuracy=-1


model_save_path = os.path.join('.','data','weights',f'pytorchmodel_weights_{end_epoch}.h5')
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose= True, min_lr=1e-6)
if checkpoint:
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    start_epoch=torch.load(checkpoint)['epoch']
 
 
 
for epoch in range(end_epoch):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model

        # Forward pass
        outputs =  outputs=model(images.to(device))
        loss=criterion(outputs.to(device),labels.to(device))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predicted = torch.softmax(outputs,dim=1)
        _,predicted=torch.max(predicted, 1)
        f1=f1_score(labels.cpu().numpy(),predicted.cpu().numpy(),average='weighted')

    # Evaluate on the validation set
    val_accuracy,val_f1,val_loss = validate(val_loader, model, criterion, device)
    print(f'Epoch [{epoch+1}/{end_epoch}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Update learning rate
    scheduler.step(val_loss)

    # Save the model if it has the best validation accuracy so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        model_savepath = f'./model/cnn/{epoch+1}{val_accuracy}.pkl'
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved the new best model with accuracy: {best_val_accuracy:.4f} at epoch {epoch+1}")

    # Early stopping check
    early_stopper(val_loss)
    if early_stopper.early_stop:
        print("Early stopping")
        break
        
        
    
# Save the model for future use and optimization.
torch.save({
'epoch': epoch,
'state_dict': model.state_dict(),
'optimizer' : optimizer.state_dict()},
'checkpoint.epoch.1.{}.pth.tar'.format(epoch))
