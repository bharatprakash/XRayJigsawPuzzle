import os, numpy as np
from time import time
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import pandas as pd

from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

class DataLoader(data.Dataset):
    def __init__(self, data_path, txt_list, n):
        self.data_path = data_path
        #self.names, self.labels = self.__dataset_info_Panda(txt_list, n)
        #self.names, self.labels = self.__dataset_CatDog(txt_list, n)
        self.names, self.labels = self.__dataset_Multi(txt_list, n)
        self.N = len(self.names)

        self.__image_transformer = transforms.Compose([
                            transforms.Resize(225, Image.BILINEAR),
                            transforms.CenterCrop(224),
                            transforms.ToTensor()])

    def __getitem__(self, index):
        framename = self.data_path+'/'+self.names[index]

        img = Image.open(framename).convert('L')

        img = self.__image_transformer(img)
        label = self.labels[index]
        return img, torch.from_numpy(label)


    def __len__(self):
        return len(self.names)

    def __dataset_info_Panda(self, txt_labels, n):
        df = pd.read_csv(txt_labels)

        rawLabels = list(df['Finding Labels'])
        lblCtr = Counter(rawLabels)
        uniqLabels = []
        for key in lblCtr.keys():
            uniqLabels += key.split('|')
        uniqLabels = list(set(uniqLabels))
        self.classes = len(uniqLabels)
        mLblBinarizer = MultiLabelBinarizer(classes=uniqLabels)

        if (n > 0):
            df = df[:n]
        else:
            df = df[n:]

        file_names = list(df['Image Index'])
        rawLabels = list(df['Finding Labels'])
        tempLabels = []

        for lbl in rawLabels:
            label = lbl.split("|")
            tempLabels.append(label)

        labels = list(mLblBinarizer.fit_transform(tempLabels))

        #for i, label in enumerate(rawLabels):
        #    if 'Infiltration' in label:
        #        labels[i] = 1

        return file_names, labels

    def __dataset_Multi(self, txt_labels, n):

        file_names = []
        labels = []

        fp = open(txt_labels + "_img.txt", 'r')
        file_names = [x.strip() for x in fp]
        fp.close()

        labels = np.loadtxt(txt_labels + "_label.txt", dtype=np.int64)
        fp.close()

        if (n > 0):
            file_names = file_names[:n]
            labels = labels[:n]
        else:
            file_names = file_names[n:]
            labels = labels[n:]

        self.classes = 5

        return file_names, labels

    def __dataset_CatDog(self, txt_labels, n):

        file_names = []
        labels = []

        with open(txt_labels, 'r') as f:
            for line in f.readlines():
                f, i, s = line.strip().split()
                file_names.append(f)
                labels.append(int(i) - 1)

        filename = np.array(file_names)
        labels = np.array(labels)

        if (n > 0):
            file_names = file_names[:n]
            labels = labels[:n]
        else:
            file_names = file_names[n:]
            labels = labels[n:]

        return file_names, labels
