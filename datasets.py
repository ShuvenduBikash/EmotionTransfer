import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CelebADataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train', attributes=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']):
        # celebA: root='D:\Research\\data\\CelebA\\img_align_celeba'
        self.transform = transforms.Compose(transforms_)

        self.selected_attrs = attributes
        self.files = sorted(glob.glob('%s/*.jpg' % root))
        self.files = self.files[:-2000] if mode == 'train' else self.files[-2000:]
        self.label_path = glob.glob('%s/*.txt' % root)[0]
        self.annotations = self.get_annotations()

    def get_annotations(self):
        """Extracts annotations for CelebA"""
        annotations = {}
        lines = [line.rstrip() for line in open(self.label_path, 'r')]
        self.label_names = lines[1].split()

        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == '1'))
            annotations[filename] = labels

        return annotations

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split('\\')[-1]
        img = self.transform(Image.open(filepath))
        label = self.annotations[filename]
        label = torch.FloatTensor(np.array(label))

        return img, label

    def __len__(self):
        return len(self.files)
    


class UTKFaceDataset(Dataset):
    def __init__(self, root='D:\\Research\\data\\UTKFace', transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob('%s/*.jpg' % root))



    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        
        filename = filepath.split('\\')[-1]
        img = self.transform(Image.open(filepath))
        class_= int(filename.split('_')[0])
        
        label = torch.zeros(8)
        if class_<=12:
            label[0]=1
        if 13<=class_<=20:
            label[1]=1
        if 21<=class_<=25:
            label[2]=1
        if 26<=class_<=30:
            label[3]=1
        if 31<=class_<=39:
            label[4]=1
        if 40<=class_<=49:
            label[5]=1
        if 50<=class_<=64:
            label[6]=1
        if class_>64:
            label[7]=1

        return img, label

    def __len__(self):
        return len(self.files)
    

