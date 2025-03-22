#Runs the full pipeline
import os

# Run Training
os.system("python train.py")

# Run Testing
os.system("python test.py")

# Convert Model to ONNX
os.system("python export.py")
