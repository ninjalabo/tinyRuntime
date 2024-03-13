# Don't edit this file! This was automatically generated from "model.ipynb".

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # swap the output layer and freeze all other layers
        for p in self.model.parameters():
          p.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
