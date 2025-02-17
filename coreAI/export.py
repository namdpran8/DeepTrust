#Converts to ONNX for deployment

import torch
from model import get_model

# Load Model
model = get_model()
model.load_state_dict(torch.load("deepfake_detector.pth"))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(model, dummy_input, "deepfake_detector.onnx", opset_version=11)
print("Model exported to ONNX format.")
