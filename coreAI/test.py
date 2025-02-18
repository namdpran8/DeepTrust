# This Python script is handling the evaluation of a deep learning model for detecting deepfake
# images. Here is a breakdown of what each part of the script is doing:
# This Python script is handling the evaluation of a deep learning model for detecting deepfake
# images. Here is a breakdown of what each part of the script is doing:
#Handles model evaluation   
import torch
from dataset import get_dataloader
from model import get_model
# The code snippet you provided is responsible for loading the test data and the trained deep learning
# model for evaluation.

# Load Test Data
test_loader = get_dataloader("test_dataset", batch_size=32, shuffle=False)

# Load Model
model = get_model()
model.load_state_dict(torch.load("deepfake_detector.pth"))
model.eval()

# Evaluate Model
# This part of the script is evaluating the deep learning model's performance on a test dataset.

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images).squeeze()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
