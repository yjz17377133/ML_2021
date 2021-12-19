import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pickle
import copy

class CarDataset_train(data.Dataset):
    """
    person attribute dataset interface
    """
    def __init__(
        self, 
        split,
        train=False,
        transform=None,
        **kwargs):
        self.train_file = 'dataset/cars_train_crop'
        self.classes_name = np.load('dataset/meta.npy')
        self.all_image = np.load('dataset/train_image.npy')
        self.all_label = np.load('dataset/train_label.npy')
        self.all_label = self.all_label - 1
        self.transform = transform
        mid = split
        self.train_image = self.all_image[0:mid]
        self.val_image = self.all_image[mid:-1]
        self.train_label = self.all_label[0:mid]
        self.val_label = self.all_label[mid:-1]
        self.image, self.label = (self.train_image, self.train_label) if train else (self.val_image, self.val_label)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the index of the target class
        """
        imgname, target = self.image[index], self.label[index]
        # load image and labels
        imgname = os.path.join(self.train_file, imgname)
        img = Image.open(imgname)
        if self.transform is not None:
            img = self.transform( img )
        
        # default no transform
        target = np.array(target).astype(np.int)

        return img, target

    # useless for personal batch sampler
    def __len__(self):
        return len(self.image)

class CarDataset_test(data.Dataset):
    """
    person attribute dataset interface
    """
    def __init__(
        self, 
        transform=None,
        **kwargs):
        self.test_file = 'dataset/cars_test_crop'
        self.classes_name = np.load('dataset/meta.npy')
        self.test_image = np.load('dataset/test_image.npy')
        self.transform = transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the index of the target class
        """
        imgname = self.test_image[index]
        # load image and labels
        imgname = os.path.join(self.train_file, imgname)
        img = Image.open(imgname)
        if self.transform is not None:
            img = self.transform( img )
        

        return img

    # useless for personal batch sampler
    def __len__(self):
        return len(self.image)