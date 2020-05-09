"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
import model

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'team_name'
    team_number = 1
    round_number = 1
    team_member = []
    contact_email = '@nyu.edu'

    def __init__(self, model_file='model'):
        self.model = model.Model()
        self.model.load_state_dict(torch.load('model_sd'))
        self.model.cuda()

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        masks = self.model(samples)
        bboxes = tuple(self.model.bounding_boxes(mask) for mask in masks)
        return bboxes

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        masks = self.model(samples)
        roadmaps = torch.stack([self.model.binary_roadmap(mask) for mask in masks])
        return roadmaps
