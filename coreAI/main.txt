import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import timm  # For EfficientNet


# Load the model
# Load EfficientNet-B3 (or any variant)
efficientnet = timm.create_model('efficientnet_b3', pretrained=True)

# Modify the classifier layer
efficientnet.classifier = nn.Identity()  # Removes last FC layer


# Load MobileNetV3-Large
mobilenet = models.mobilenet_v3_large(pretrained=True)
mobilenet.classifier = nn.Identity()  # Remove last layer


# Freeze base layers (optional)
for param in efficientnet.parameters():
    param.requires_grad = False

for param in mobilenet.parameters():
    param.requires_grad = False



class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.efficientnet = efficientnet
        self.mobilenet = mobilenet
        
        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(3072, 512),  # 1536 (EfficientNet) + 1536 (MobileNet)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),  # Binary Classification (Fake/Real)
            nn.Sigmoid()
        )

    def forward(self, x):
        eff_out = self.efficientnet(x)
        mob_out = self.mobilenet(x)
        combined = torch.cat((eff_out, mob_out), dim=1)  # Concatenate features
        return self.fc(combined)

# Initialize model
model = DeepfakeDetector().cuda()
model.load_state_dict(torch.load('model.pth'))

# Loss and Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy for fake vs real
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)

# Training Function
def train(model, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.float().cuda()

            # Forward
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Assume train_loader is your DataLoader with deepfake images

