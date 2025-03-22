# This Python script is performing the following tasks:
import torch
from model import get_model

# Load Model
# The code snippet `model = get_model()` is creating an instance of a neural network model using a
# function called `get_model()`. This function likely defines the architecture of the model and
# initializes it with certain parameters.
model = get_model()
model.load_state_dict(torch.load("deepfake_detector.pth"))
model.eval()
# This code snippet is exporting a PyTorch model to the ONNX format. Here's a breakdown of what each
# line is doing:

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(model, dummy_input, "deepfake_detector.onnx", opset_version=11)
print("Model exported to ONNX format.")
