import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, data_path, img_size=416):
        self.data_path = data_path
        self.img_size = img_size
        self.classes = self._get_classes()
        self.img_files, self.label_files = self._get_files()

    def __getitem__(self, index):
        img_path = self.img_files[index]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img) / 255.0
        img = torch.from_numpy(img).float().permute(2, 0, 1)

        label_path = self.label_files[index]
        label = np.loadtxt(label_path)
        label = torch.from_numpy(label).float()

        return img, label

    def __len__(self):
        return len(self.img_files)

    def _get_classes(self):
        classes_path = os.path.join(self.data_path, 'classes.names')
        with open(classes_path, 'r') as f:
            classes = f.read().split('\n')[:-1]
        return classes

    def _get_files(self):
        img_files = glob.glob(os.path.join(self.data_path, 'images', '*.jpg'))
        label_files = [path.replace('images', 'labels').replace('.jpg', '.txt') for path in img_files]
        return img_files, label_files
