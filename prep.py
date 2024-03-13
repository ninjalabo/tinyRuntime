# Don't edit this file! This was automatically generated from "prep.ipynb".

import os
import shutil
import torchvision
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from export import serialize_fp32

base = "data"
os.makedirs(base, exist_ok=True)

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
download_and_extract_archive(url, download_root=base, extract_root=base)

# resize images into (3, 224, 224) and normalize
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
dataset = ImageFolder(root="data/imagenette2/val", transform=transform)

# save transformed images to binary files
dst = "data/imagenette2/val_transformed"
os.makedirs(dst, exist_ok=True)

for i, (image, label) in enumerate(dataset):
    if i % 500 == 0:
        print(f"Wrote {i}/{len(dataset)}")

    dsd = os.path.join(dst, str(int(label)))
    os.makedirs(dsd, exist_ok=True)
    df = os.path.join(dsd, str(i))
    f = open(df, "wb")
    serialize_fp32(f, image)
    f.close()
print("Done")

# create directory containing the subset of `val_transformed`
src = "data/imagenette2/val_transformed"
dst = "data/imagenette2/val_transformed_subset"
os.makedirs(dest, exist_ok=True)
seed = 4
nsamples = 10

for label in range(10):
    ssd = os.path.join(src, str(label))
    dsd = os.path.join(dst, str(label))
    os.makedirs(dsd, exist_ok=True)

    # shuffle image files
    files = os.listdir(ssd)
    random.Random(seed).shuffle(files)

    # copy 10 samples to `val_transformed_subset`
    for f in files[:nsamples]:
        sf = os.path.join(ssd, f)
        df = os.path.join(dsd, f)
        shutil.copy(sf, df)
