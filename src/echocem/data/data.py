import torch
import torchvision
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F

import pandas as pd
import numpy as np

import os
from pathlib import Path

from functools import lru_cache
import threading
from collections import OrderedDict

# Mean: -0.2745382931759886, std: 28.8
default_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.2745], std=[28.8])
])

def filter_list(list, fnc):
    return [item for item in list if fnc(item)]

class CemDataset(Dataset):
    """
    Optimized dataset class for loading ultrasound images and annotations.
    - Caching of loaded images
    - Precomputation of valid indices
    - Memory-efficient preprocessing
    - Optional preloading of all data
    """
    def __init__(self, 
                 images_path, 
                 annotations_path=None, 
                 transform=default_transform, 
                 crop_size=(256, 272), 
                 excluded_wells=[],
                 preload_data=True,
                 file_format='processed',
                 cache_size=100):
        
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.transform = transform
        self.crop_size = crop_size
        self.preload_data = preload_data
        self.cache_size = cache_size
        
        # Pre-filter and sort image paths
        if len(excluded_wells) > 0:
            if file_format == 'processed':
                self.images = sorted([f for f in os.listdir(images_path) 
                                if int(f.split('_')[0]) not in excluded_wells])
            else:
                    self.images = sorted([f for f in os.listdir(images_path)
                                            if int(f.split('_')[1]) not in excluded_wells])
        else:
            self.images = sorted([f for f in os.listdir(images_path)])
        
        if self.annotations_path:
            if len(excluded_wells) > 0:
                if file_format == 'processed':
                    self.annotations = sorted([f for f in os.listdir(annotations_path) 
                                            if int(f.split('_')[0]) not in excluded_wells])
                else:
                        self.annotations = sorted([f for f in os.listdir(annotations_path)
                                                if int(f.split('_')[1]) not in excluded_wells])
            else:
                self.annotations = sorted([f for f in os.listdir(annotations_path)])
            if len(self.images) != len(self.annotations):
                raise ValueError("Number of images and annotations must be equal.")
        else:
            self.annotations = None
            
        if len(self.images) == 0:
            raise ValueError("No images found.")
            
        # Initialize cache as OrderedDict for FIFO behavior
        self.cache = OrderedDict()
        
        # Preload data if requested
        if self.preload_data:
            print("Preloading data into memory...")
            self._preload_data()

    def _preload_data(self):
        """Preload all data into memory"""
        for idx in range(len(self.images)):
            image_path = os.path.join(self.images_path, self.images[idx])
            self.cache[idx] = {
                'image': torch.from_numpy(np.load(image_path)).float()
            }
            
            if self.annotations_path:
                annotation_path = os.path.join(self.annotations_path, self.annotations[idx])
                self.cache[idx]['annotation'] = torch.from_numpy(np.load(annotation_path)).long()

    def _load_item(self, idx):
        """Load and cache a single item"""
        if idx in self.cache:
            return self.cache[idx]
        
        image_path = os.path.join(self.images_path, self.images[idx])
        item = {
            'image': torch.from_numpy(np.load(image_path)).float()
        }
        
        if self.annotations_path:
            annotation_path = os.path.join(self.annotations_path, self.annotations[idx])
            item['annotation'] = torch.from_numpy(np.load(annotation_path)).long()
            
        # Manage cache size using FIFO
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # Remove oldest item
        self.cache[idx] = item
            
        return item

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data (either from cache or disk)
        item = self._load_item(idx) if not self.preload_data else self.cache[idx]
        image = item['image']
        annotation = item.get('annotation', None)

        # Add channel dimension if needed
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if annotation is not None and annotation.ndim == 2:
            annotation = annotation.unsqueeze(0)

        # Apply random crop
        if self.crop_size is not None:
            i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
            image = F.crop(image, i, j, h, w)
            if annotation is not None:
                annotation = F.crop(annotation, i, j, h, w)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return (image, annotation) if annotation is not None else image


if __name__ == "__main__":
    # Example usage:
    root_directory = Path(os.getcwd())
    images_path = root_directory / Path("data/processed/X_train/images")
    annotations_path = root_directory / Path("data/processed/X_train/annotations")
    dataset = CemDataset(images_path, annotations_path, excluded_wells=[3, 4])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    next(iter(dataloader))
    # Without the annotations
    dataset = CemDataset(images_path, excluded_wells=[3, 4])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    next(iter(dataloader))