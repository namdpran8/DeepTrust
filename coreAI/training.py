# This Python script is for training a deep learning model using PyTorch. Here's a breakdown of what
# the code does:
# This Python script is for training a deep learning model using PyTorch. Here's a breakdown of what
# the code does:
#Handles training
# The code snippet you provided is a Python script for training a deep learning model using PyTorch.

import torch
from tqdm import tqdm
from dataset import get_dataloader
from model import get_model
from torch.optim import Adam
import torch
import torch.nn as nn


# Load Dataset & Model
train_loader = get_dataloader("dataset", batch_size=32)
model = get_model()
print("Dataset and Model loaded successfully." + str(model))


# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.0001)
print("Loss function and optimizer defined successfully."+ str(criterion))

# This part of the code is the training loop for a deep learning model. Here's a breakdown of what it
# does:

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0


    for images, labels in tqdm(train_loader):
        images, labels = images.cuda(), labels.float().cuda()

        optimizer.zero_grad()
        outputs = model(images).squeeze()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "deepfake_detector.pth")
print("Model saved successfully.")