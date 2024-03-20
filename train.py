# Don't edit this file! This was automatically generated from "train.ipynb".

import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
