# Don't edit this file! This was automatically generated from "prep.ipynb".

import os
import shutil
import random
import torchvision
from torchvision.datasets.utils import download_and_extract_archive as __download_and_extract_archive
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from export import serialize_fp32

data_root = 'data'
imagenette2_root = os.path.join(data_root, 'imagenette2')

def download_and_extract_archive(root=data_root):
    os.makedirs(root, exist_ok=True)
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    __download_and_extract_archive(url, download_root=root, extract_root=root)

def find_files(root:str=data_root, ext:str='jpeg') -> list[str]:
    ext = '.' + ext if len(ext) else ''
    l = [os.path.join(d, o) for d, _, files in os.walk(root) for o in files if o.lower().endswith(ext)]
    return l

dls = ImageFolder(os.path.join(imagenette2_root, 'val'),
                  transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
val_tx_root = os.path.join(imagenette2_root, 'val_transformed')

import matplotlib.pyplot as plt
import torch
import struct

def show_image(label, number):
    path = os.path.join(val_tx_root, str(label), str(number))
    with open(path, "rb") as f:
        sizeof_float, nch, h, w = 4, 3, 224, 224
        image = torch.tensor(struct.unpack("f"*(nch*h*w), f.read(sizeof_float*nch*h*w))).view(nch,h,w)

    # imshow accepts image shape (height, width, nch)
    image_transposed = image.permute(1, 2, 0) 
    plt.imshow(image_transposed)
    print(f"Image shape (nch, h, w): {(nch, h, w)} {path}")

def sample_files(src, dst, n=10, seed=4):
    """create directory containing the subset of `src`"""
    shutil.rmtree(dst, ignore_errors=True)
    subdirs = [o for o in os.listdir(src) if os.path.isdir(os.path.join(src, o))]
    for o in subdirs:
        files = os.listdir(os.path.join(src, o))
        random.Random(seed).shuffle(files)
        for f in files[:n]:
            os.makedirs(os.path.join(dst, o), exist_ok=True)
            shutil.copy(os.path.join(src, o, f), os.path.join(dst, o, f))
