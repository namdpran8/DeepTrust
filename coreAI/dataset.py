#Handles data loading & preprocessing
# The code snippet you provided is importing necessary libraries and modules for handling data loading
# and preprocessing tasks in a Python script. 
import os
import cv2
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import os

# The code snippet you provided is specifying the root directory path as
# `C:\hackathon\DeepTrust\coreAI` and another path as `C:\hackathon\DeepTrust\coreAI\Dataset`. It then
# uses the `os.listdir()` function to get the list of files and directories present in the `root_dir`
# directory. Finally, it prints out the list of files and directories obtained from the `root_dir`
# directory.
# Specify the path
root_dir = r"C:\hackathon\DeepTrust\coreAI"
path=r"C:\hackathon\DeepTrust\coreAI\Dataset"
# Get the list of files and directories in the specified path
files = os.listdir(root_dir)

# Print the list
print(files)
# The code snippet you provided is defining a series of image augmentations using the Albumentations
# library in Python. 

# Define Image Augmentations
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

print("Image Augmentations defined successfully.")

# Define Dataset Class
class DeepfakeDataset(Dataset):
    """
    The code defines a custom dataset class for a deepfake dataset and a function to create a DataLoader
    using this dataset.
    
    :param root_dir: The `root_dir` parameter in the `DeepfakeDataset` class and the `get_dataloader`
    function refers to the root directory where your dataset is stored. This is the main directory
    containing subdirectories for different categories or classes of data (in this case, "real" and
    "fake"
    :param batch_size: The `batch_size` parameter in the `get_dataloader` function specifies the number
    of samples that will be propagated through the neural network at a time during training. It is a
    hyperparameter that can affect the training process and memory usage, defaults to 32 (optional)
    :param shuffle: The `shuffle` parameter in the `get_dataloader` function determines whether the data
    will be shuffled at every epoch before creating batches. If `shuffle=True`, the data will be
    shuffled, which can be useful to introduce randomness and prevent the model from learning the order
    of the data. If `shuffle, defaults to True (optional)
    :return: The code defines a custom dataset class `DeepfakeDataset` that loads images from the
    specified `root_dir` directory, assigns labels based on the subdirectories ("real" and "fake"), and
    applies transformations if provided. The `__len__` method returns the total number of images in the
    dataset, and the `__getitem__` method loads and processes individual images along with their labels.
    """
    def __init__(self, root_dir, transform=None, path="C:\hackathon\DeepTrust\coreAI"):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        for label, category in enumerate(["real", "fake"]):
            path = os.path.join(root_dir, category)
            for img_name in os.listdir(path):
                self.data.append(os.path.join(path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, torch.tensor(label, dtype=torch.long)
print("Dataset class defined successfully.")

# Function to get DataLoader
def get_dataloader(root_dir, batch_size=32, shuffle=True):
    dataset = DeepfakeDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
print("DataLoader created successfully.")
