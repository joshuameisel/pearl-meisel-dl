import os
import random

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200
from data_helper import UnlabeledDataset, LabeledDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from data_helper import *
from helper import collate_fn, draw_box
import model
import torch.optim as optim

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = 'data'
annotation_csv = 'data/annotation.csv'

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

class MaskDataset(LabeledDataset):
    def __getitem__(self, index):
        sample, target, road_image = LabeledDataset.__getitem__(self, index)
        road_image = road_image.int()
        
        for i, bb in enumerate(target['bounding_box']):
            point_squence = torch.stack([bb[:, 0], bb[:, 1], bb[:, 3], bb[:, 2], bb[:, 0]])
            x = (point_squence.T[0] * 10 + 400).int()
            y = (-point_squence.T[1] * 10 + 400).int()

            bottom = y.min()
            top = y.max()
            left = x.min()
            right = x.max()

            road_image[bottom:top,left:right] = target['category'][i]

        return sample, road_image.long()
    
# transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
transform = torchvision.transforms.ToTensor()

dataset = MaskDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index,
                                  transform=transform,
                                  extra_info=False
                                 )
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = model.Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net.to(device)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        print('hello')
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if True: #i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1))# 100))
            running_loss = 0.0

print('Finished Training')