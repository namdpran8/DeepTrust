#Defines the DeepfakeDetector model
import torch
import torch.nn as nn
import timm  # Pretrained models

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.efficientnet = timm.create_model("efficientnet_b0", pretrained=True)
        self.mobilenet = timm.create_model("mobilenetv3_large_100", pretrained=True)

        # Remove last classifier layers
        self.efficientnet.classifier = nn.Identity()
        self.mobilenet.classifier = nn.Identity()

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1280 + 1280, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        eff_out = self.efficientnet(x)
        mob_out = self.mobilenet(x)
        x = torch.cat((eff_out, mob_out), dim=1)
        x = self.fc(x)
        return x

# Function to get model
def get_model():
    return DeepfakeDetector().cuda()
