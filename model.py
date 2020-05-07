import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
from scipy.ndimage.measurements import label

class Model(nn.Module):
    def __init__(self):
        super().__init__()
#         self.fcns = tuple(torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=11) for i in range(6))
        self.fcn1 = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=11)
        self.fcn2 = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=11)
        self.fcn3 = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=11)
        self.fcn4 = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=11)
        self.fcn5 = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=11)
        self.fcn6 = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=11)
        
        self.conv = nn.Conv2d(66, 11, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(800, 800))

    def forward(self, x):
        x1 = self.fcn1(x[:,0,:,:,:])['out']
        x2 = self.fcn1(x[:,1,:,:,:])['out']
        x3 = self.fcn1(x[:,2,:,:,:])['out']
        x4 = self.fcn1(x[:,3,:,:,:])['out']
        x5 = self.fcn1(x[:,4,:,:,:])['out']
        x6 = self.fcn1(x[:,5,:,:,:])['out']
        
#         x = tuple(self.fcns[i](x[:,i,:,:,:])['out'] for i in range(6))
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        x = self.conv(x)
        x = self.upsample(x)
        
        return x

    # need to properly format results and convert to coordinate space
    def bounding_boxes(self, roadmask):
        results = []
        cats = [x for x in np.unique(roadmask) if x > 1]
        for c in cats:
            cmask = roadmask == c
            labeled, n = label(cmask)
            for i in range(1, n+1):
                y, x = np.where(labeled == i) 
                y, x = ((y - 400) / 10), ((x - 400) / 10)
                y, x = [y[0], y[-1]], [x[0], x[-1]]
                (top, bottom), (left, right) = y, x
                results.append([[top, top, bottom, bottom], [left, right, left, right]])
        return results
    
    def binary_roadmap(self, roadmask):
        return roadmask > 0