#Handles model evaluation   
import torch
from dataset import get_dataloader
from model import get_model

# Load Test Data
test_loader = get_dataloader("test_dataset", batch_size=32, shuffle=False)

# Load Model
model = get_model()
model.load_state_dict(torch.load("deepfake_detector.pth"))
model.eval()

# Evaluate Model
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images).squeeze()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
