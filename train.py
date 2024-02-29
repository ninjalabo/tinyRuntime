# Don't edit this file! This was automatically generated from "train.ipynb".

import numpy as np
import torchvision
from torchvision import transforms, datasets
import torch
from torch import nn
import matplotlib.pyplot as plt
import struct
import os

import torchvision.models as models
from torchvision.models import ResNet18_Weights
from model import ResNetMnist # my model
from export import export_model
from export import export_modelq8

def generate_dataloaderImagenette(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),        
        transforms.ToTensor(),             
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    #data = torchvision.datasets.Imagenette("./data", download=True)
    traindataset = datasets.ImageFolder(root='./data/imagenette2/train', transform=transform)
    testdataset = datasets.ImageFolder(root='./data/imagenette2/val', transform=transform)

    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

def generate_dataloaderMNIST(batch_size=32):
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

        # Fetch the first batch
        for inputs, targets in testloader:
            outputs = model(inputs)
            vloss += loss_fn(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted==targets).sum().item()
    
    return vloss/len(testloader),  correct/len(testloader.dataset)


def train_model(model):  
    # training
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainloader, testloader = generate_dataloaderMNIST()

    for epoch in range(1):

        model.train()
        tloss = 0
        for inputs, targets in trainloader:
            opt.zero_grad()
            out = model(inputs)
            loss = loss_fn(out, targets)
            loss.backward()
            tloss += loss.item()
            opt.step()

        tloss = tloss/len(trainloader)
        vloss, correct = test_model(model, testloader)

        print('LOSS train {} valid {} accuracy {:.5f}'.format(tloss, vloss, correct))
    torch.save(model, "modelMNIST.pt")
