import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
from scipy.ndimage.measurements import label

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = torchvision.models.segmentation.fcn_resnet50(pretrained=False)
        # also load pretrained feature extractor
        # also create final combining layer

    def forward(self, x):
        return self.fcn(x)

    # need to properly format results and convert to coordinate space
    def bounding_boxes(self, roadmask):
        results = []
        cats = [x for x in np.unique(roadmask) if x > 1]
        for c in cats:
            cmask = roadmask == c
            labeled, n = label(cmask)
            for i in range(1, n+1):
                y, x = np.where(labeled == i)
                y, x = [y[0], y[-1]], [x[0], x[-1]]
                (top, bottom), (left, right) = y, x
                results.append(((top, top, bottom, bottom), (left, right, left, right)))
        return results