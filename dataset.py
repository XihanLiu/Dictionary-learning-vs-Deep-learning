# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class Face(data.Dataset):
    
    def __init__(self,image_root,list_root,transforms=None,train=True):
        '''
        Get images, divide into train/val set
        '''

        self.train = train
        self.list_root = list_root
        self.image_root = image_root

        self._read_txt_file()
    
        # if transforms is None:
        #     normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])

        if not train:
            self.transforms = T.Compose([
                T.Resize([224,224]),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5)
                ])
        else:
            self.transforms = T.Compose([
                T.Resize([256,256]),
                T.RandomResizedCrop([224,224]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5)
                ])
                
    def _read_txt_file(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = self.list_root + r"\train.txt"
        else:
            txt_file = self.list_root + r"\test.txt"

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                if '.bad' in item[0]:
                    continue;
                else:
                    self.images_path.append(item[0])
                    self.images_labels.append(int(item[1])-1)


    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = self.image_root+'\\'+self.images_path[index]
        label = self.images_labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, int(label)
    
    def __len__(self):
        return len(self.images_path)
