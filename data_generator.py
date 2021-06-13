import os.path as ops
import numpy as np
import torch
import os
import cv2
import torchvision

class APPLE(torch.utils.data.Dataset):
    def __init__(self, resize=(128, 64)):
        self.train_file = 'image_paths.txt'
        self.resize = resize
        self.image_paths = []

        with open(self.train_file, 'r') as file:
            data = file.readlines()
            for l in data:
                self.image_paths.append(l)        


    def __len__(self):
        return(len(self.image_paths))
        

    def __getitem__(self, idx):        
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, dsize=self.resize, interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        image = torch.tensor(image, dtype=torch.float)
        gt = torch.tensor([1.0])
        return image, gt