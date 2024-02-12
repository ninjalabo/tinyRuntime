# Don't edit this file! This was automatically generated from "model.ipynb".

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, dim=128):
        super(Model, self).__init__()
        self.dim = dim
        self.nclass = 10
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([nn.Linear(28*28, dim), 
                                    nn.Linear(dim, dim//2)])
        self.activation = nn.ReLU()
        self.out = nn.Linear(dim//2, self.nclass)
        
    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out(x)
        return x
