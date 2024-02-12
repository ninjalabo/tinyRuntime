# Don't edit this file! This was automatically generated from "train.ipynb".

import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch import nn
import matplotlib.pyplot as plt
import struct
import os

from model import Model # my model
from export import export_model
from export import export_modelq8

def generate_dataloader(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    trainset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

def test_model(model, testloader):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        vloss = 0.
        correct = 0.
        for X,y in testloader:
            out = model(X)
            vloss += loss_fn(out, y).item()
            correct += (torch.argmax(out, 1)==y).float().sum()
    
    return vloss/len(testloader),  correct/len(testloader.dataset)

def train_model(model):  
    # training
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainloader, testloader = generate_dataloader()

    for epoch in range(3):

        model.train()
        tloss = 0
        for X,y in trainloader:
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            tloss += loss.item()
            opt.step()

        tloss = tloss/len(trainloader)
        vloss, correct = test_model(model, testloader)

        print('LOSS train {} valid {} accuracy {:.5f}'.format(tloss, vloss, correct))
