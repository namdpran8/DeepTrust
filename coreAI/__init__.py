# __init__.py

print("Initializing the coreAI package.")

# You can also import other files from the package if needed:
from .dataset import DataLoader
from .model import DeepfakeDetector
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Specify the path
path = r'C:\hackathon\DeepTrust\coreAI\dataset'

# Get the list of files and directories in the specified path
files = os.listdir(path)

# Print the list
print(files)
