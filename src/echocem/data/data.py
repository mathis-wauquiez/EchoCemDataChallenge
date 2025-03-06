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
import random

# Mean: -0.2745382931759886, std: 28.8
default_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.2745], std=[28.8]),
])

def filter_list(list, fnc):
    return [item for item in list if fnc(item)]


class CemDataset(Dataset):

    def __init__(self,
                 images_path,
                 annotations_path=None,
                 transform=default_transform,
                 crop_size=(272, 272),
                 excluded_wells=[],
                 file_format='processed'
    ):
        
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.transform = transform
        self.crop_size = crop_size
        self.excluded_wells = excluded_wells
        self.file_format = file_format
        
        # Loading the files

        # 'well_n_section_m_patch_l.npy' for raw files / 'n_m' for processed files
        well_idx = 0 if file_format == 'processed' else 1


        # Load the sensor images
        self.images_paths = sorted([f for f in os.listdir(images_path) 
                        if int(f.split('_')[well_idx]) not in excluded_wells])
        self.images_npy = [np.load(os.path.join(images_path, f)) for f in self.images_paths]


        # Load the annotations
        if self.annotations_path:

            self.annotations_paths = sorted([f for f in os.listdir(annotations_path) 
                                    if int(f.split('_')[well_idx]) not in excluded_wells])
            
            self.annotations_npy = [np.load(os.path.join(annotations_path, f)) for f in self.annotations_paths]

            if len(self.images_paths) != len(self.annotations_paths):
                raise ValueError("Number of images and annotations must be equal.")
            
        else:
            self.annotations_paths = None

    def get_number_of_crops(self, image_height):
        # Return the number of crops that can be extracted from the image
        return image_height - self.crop_size[0] + 1 if self.crop_size is not None else 1
    
    def __len__(self):
        # Number of crops in all images
        return sum([self.get_number_of_crops(image.shape[0]) for image in self.images_npy]) if self.crop_size is not None else len(self.images_npy)
    
    def __getitem__(self, idx):
        # Get the image index and the crop index
        if self.crop_size is None:
            image_idx = idx
            idx = 0
        
        else:
            image_idx = 0
            for image in self.images_npy:
                number_of_crops = self.get_number_of_crops(image.shape[0])
                if idx < number_of_crops:
                    break
                idx -= number_of_crops
                image_idx += 1
        
        # Choose the right image and annotation
        image = self.images_npy[image_idx]
        annotation = self.annotations_npy[image_idx] if self.annotations_paths else None
        
        # Crop the image
        if self.crop_size is not None:
            image = image[idx:idx+self.crop_size[0], :]
            if annotation is not None:
                annotation = annotation[idx:idx+self.crop_size[0], :]
        
        # Add channel dimension
        if image.ndim == 2:
            image = image[np.newaxis, :, :]

        

        # Apply random vertical flip
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            if annotation is not None:
                annotation = np.flip(annotation, axis=0).copy() # raison du .copy(): https://discuss.pytorch.org/t/negative-strides-in-tensor-error/134287/2
        
        image = torch.from_numpy(image).float()

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        print(image.shape) # Should be (C, H, W)

        annotation = torch.from_numpy(annotation).long() if annotation is not None else None
        return (image, annotation) if annotation is not None else image

    def __reduce__(self):
        """
        Pickling method for the dataset. (required for multiprocessing with DataLoader)
        """
        # Return a tuple of class, constructor arguments, and additional state
        return (self.__class__, 
                (self.images_path, self.annotations_path, self.transform, 
                 self.crop_size, self.excluded_wells, self.file_format), 
                {})


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