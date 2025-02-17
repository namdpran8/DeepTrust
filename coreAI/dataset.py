
#Handles data loading & preprocessing#

import os
import cv2
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

# Define Image Augmentations
transform = A.Compose([
    A.Resize(224, 224),
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Define Dataset Class
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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

# Function to get DataLoader
def get_dataloader(root_dir, batch_size=32, shuffle=True):
    dataset = DeepfakeDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
