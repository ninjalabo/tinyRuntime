# Don't edit this file! This was automatically generated from "test_dataset_generator.ipynb".

def denormalize(x, stats=imagenet_stats):
    mean, std = [torch.tensor(o).view(3,1,1) for o in stats]
    return x * std + mean

def BinImageCreate(fn, stats=imagenet_stats):
    with open(fn, "rb") as f:
        x = struct.unpack(f'{3*224*224}f', f.read())
    x = torch.tensor(x).view(3, 224, 224)
    return x
