import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CelebADataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train',
                 attributes=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']):
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


class KDEFDataset(Dataset):
    def __init__(self, root='data\\KDEF', transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = []

        for _ in range(5):
            for path, subdirs, files in os.walk(root):
                for name in files:
                    self.files.append(os.path.join(path, name))

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]

        filename = filepath.split('\\')[-1]
        img = self.transform(Image.open(filepath))
        class_ = filename[4:6]

        label = torch.zeros(7)
        if class_ == 'NE':
            label[0] = 1
        if class_ == 'AF':
            label[1] = 1
        if class_ == 'AN':
            label[2] = 1
        if class_ == 'DI':
            label[3] = 1
        if class_ == 'SU':
            label[4] = 1
        if class_ == 'SA':
            label[5] = 1
        if class_ == 'HA':
            label[6] = 1

        return img, label

    def __len__(self):
        return len(self.files)


class KDEF_NE_Dataset(Dataset):
    def __init__(self, root='data\\KDEF', transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = []

        for path, subdirs, files in os.walk(root):
            for name in files:
                self.files.append(os.path.join(path, name))
        self.selected_files = []

        for file in self.files:
            filename = file.split('\\')[-1]
            class_ = filename[4:6]
            if class_ == 'NE':
                self.selected_files.append(file)

        self.files = self.selected_files

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]

        filename = filepath.split('\\')[-1]
        img = self.transform(Image.open(filepath))
        class_ = filename[4:6]

        label = torch.zeros(7)
        if class_ == 'NE':
            label[0] = 1
        if class_ == 'AF':
            label[1] = 1
        if class_ == 'AN':
            label[2] = 1
        if class_ == 'DI':
            label[3] = 1
        if class_ == 'SU':
            label[4] = 1
        if class_ == 'SA':
            label[5] = 1
        if class_ == 'HA':
            label[6] = 1

        return img, label

    def __len__(self):
        return len(self.files)


class CustomDataset(Dataset):
    def __init__(self, rootList=['D:\\Research\\data\\KDEF_and_AKDEF\\KDEF', 'D:\\Research\\data\\DDCFL'],
                 transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = []

        for root in rootList:
            for path, subdirs, files in os.walk(root):
                for name in files:
                    self.files.append(os.path.join(path, name))

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]

        filename = filepath.split('\\')[-1]
        img = self.transform(Image.open(filepath))

        if len(filename) > 13:
            class_ = filename.split('_')[2]
        else:
            class_ = filename[4:6]

        label = torch.zeros(7)
        if class_ == 'NE' or class_ == 'Neutral':
            label[0] = 1
        if class_ == 'AF' or class_ == 'Afraid':
            label[1] = 1
        if class_ == 'AN' or class_ == 'Angry':
            label[2] = 1
        if class_ == 'DI' or class_ == 'Disgusted':
            label[3] = 1
        if class_ == 'SU' or class_ == 'Surprised':
            label[4] = 1
        if class_ == 'SA' or class_ == 'Sad':
            label[5] = 1
        if class_ == 'HA' or class_ == 'Happy':
            label[6] = 1

        return img, label

    def __len__(self):
        return len(self.files)


class Custom_NE_Dataset(Dataset):
    def __init__(self, rootList=['D:\\Research\\data\\KDEF_and_AKDEF\\KDEF', 'D:\\Research\\data\\DDCFL'],
                 transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = []

        for root in rootList:
            for path, subdirs, files in os.walk(root):
                for name in files:
                    self.files.append(os.path.join(path, name))
        self.selected_files = []

        for file in self.files:
            filename = file.split('\\')[-1]

            if len(filename) > 13:
                class_ = filename.split('_')[2]
            else:
                class_ = filename[4:6]
            if class_ == 'NE' or class_ == 'Neutral':
                self.selected_files.append(file)

        self.files = self.selected_files

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]

        filename = filepath.split('\\')[-1]
        img = self.transform(Image.open(filepath))

        if len(filename) > 13:
            class_ = filename.split('_')[2]
        else:
            class_ = filename[4:6]

        label = torch.zeros(7)
        if class_ == 'NE' or class_ == 'Neutral':
            label[0] = 1
        if class_ == 'AF' or class_ == 'Afraid':
            label[1] = 1
        if class_ == 'AN' or class_ == 'Angry':
            label[2] = 1
        if class_ == 'DI' or class_ == 'Disgusted':
            label[3] = 1
        if class_ == 'SU' or class_ == 'Surprised':
            label[4] = 1
        if class_ == 'SA' or class_ == 'Sad':
            label[5] = 1
        if class_ == 'HA' or class_ == 'Happy':
            label[6] = 1

        return img, label

    def __len__(self):
        return len(self.files)
