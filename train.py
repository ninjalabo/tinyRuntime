# Don't edit this file! This was automatically generated from "train.ipynb".

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import struct
import os

import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA (GPU) is available.")
else:
    device = torch.device("cpu")
device

import torchvision
from torchvision.datasets.utils import download_and_extract_archive

dir_path = "data"
os.makedirs(dir_path, exist_ok=True)

# URL for Imagenette (Full size)
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

# Download and extract the dataset
download_and_extract_archive(url, download_root=dir_path, extract_root=dir_path)

def generate_dataloader(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to fit ResNet input size
        transforms.ToTensor(),
    ])
    
    train_dataset = ImageFolder(root=f"data/imagenette2/train", transform=transform)
    test_dataset = ImageFolder(root=f"data/imagenette2/val", transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=torch.get_num_threads())
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=torch.get_num_threads())
    
    return trainloader, testloader

def test_model(model, testloader):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        vloss = 0.
        correct = 0.

        # Fetch the first batch
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            vloss += loss_fn(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted==targets).sum().item()
    
    return vloss/len(testloader),  correct/len(testloader.dataset)


def train_model(model):  
    # training
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainloader, testloader = generate_dataloader()

    for epoch in range(1):

        model.train()
        tloss = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            out = model(inputs)
            loss = loss_fn(out, targets)
            loss.backward()
            tloss += loss.item()
            opt.step()

        tloss = tloss/len(trainloader)
        vloss, correct = test_model(model, testloader)

        print('LOSS train {} valid {} accuracy {:.5f}'.format(tloss, vloss, correct))
    torch.save(model, "model.pt")
