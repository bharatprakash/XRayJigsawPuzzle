import os, numpy as np
from time import time
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import pandas as pd

from PIL import Image

class DataLoader(data.Dataset):
    def __init__(self, data_path, txt_list, n):
        self.data_path = data_path
        self.names, self.labels = self.__dataset_info_Panda(txt_list, n)
        self.N = len(self.names)

        self.__image_transformer = transforms.Compose([
                            transforms.Resize(225,Image.BILINEAR),
                            transforms.ToTensor()])

    def __getitem__(self, index):
        framename = self.data_path+'/'+self.names[index]

        img = Image.open(framename).convert('RGB')
        if np.random.rand()<0.30:
            img = img.convert('LA').convert('RGB')

        img = self.__image_transformer(img)
        label = self.labels[index]
        return img, label


    def __len__(self):
        return len(self.names)

    def __dataset_info_Panda(self, txt_labels, n):
        df = pd.read_csv(txt_labels)

        if (n > 0):
            df = df[:n]
        else:
            df = df[n:]

        file_names = list(df['Image Index'])
        labels = np.zeros(len(file_names))

        rawLabels     = list(df['Finding Labels'])
        for i, label in enumerate(rawLabels):
            if 'Pneumonia' in label:
                labels[i] = 1

        return file_names, labels
