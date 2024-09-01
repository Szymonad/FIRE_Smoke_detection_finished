
import cv2
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2
import numpy as np
import random

class FrameDataset(Dataset):
    def __init__(self, root_dir, label, transform=None, device='cpu'):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.file_paths = sorted([os.path.join(self.root_dir, file_name) for file_name in os.listdir(self.root_dir)])
        self.original_file_paths = self.file_paths.copy()  # Save the original file paths
        self.length = sum([int(cv2.VideoCapture(file_path).get(cv2.CAP_PROP_FRAME_COUNT)) for file_path in self.file_paths])
        self.current_cap = None
        self.current_frame_idx = 0
        self.device = device
        print(f'Label init: {self.label}')

    def __del__(self):
        if self.current_cap:
            self.current_cap.release()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.current_cap is None or self.current_frame_idx >= int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            if self.current_cap:
                self.current_cap.release()
            if not self.file_paths:  # If the list is empty, reset it
                self.file_paths = self.original_file_paths.copy()
            random_file_path = random.choice(self.file_paths)
            self.file_paths.remove(random_file_path)
            self.current_cap = cv2.VideoCapture(random_file_path)
            self.current_frame_idx = 0

        ret, frame = self.current_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frame = frame.to(self.device)
        self.current_frame_idx += 1

        if self.transform:
            frame = self.transform(frame)

        return frame, self.label.to(self.device)

    @staticmethod
    def prepare_transform(input_size):
        return transforms.Compose([
            v2.ToPILImage(),
            v2.Resize((input_size, input_size)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.GaussianBlur(5, sigma=(0.1, 2.0)),
            v2.RandomAdjustSharpness(3),
            v2.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.07),
            v2.RandomEqualize(0.3),
            v2.RandomPerspective(),
            v2.ElasticTransform(alpha=200.0),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    @staticmethod
    def prepare_datasets(root_dirs, labels, input_size, device='cpu'):
        transform = FrameDataset.prepare_transform(input_size)
        datasets = []
        for root_dir, label in zip(root_dirs, labels):
            if isinstance(label, torch.Tensor) and len(label.size()) == 1 and torch.all(label == label[0]):
                label = torch.tensor([label[0].item()], dtype=torch.long).to(device)
            datasets.append(FrameDataset(root_dir=root_dir, label=label, transform=transform, device=device))
        return datasets
