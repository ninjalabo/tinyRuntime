import torch
from fastai.vision.all import *

from export import Quantizer, export_model_sq8

model = torch.load("md/model.pkl", map_location="cpu").model
dls = ImageDataLoaders.from_folder(Path.cwd()/"data", valid='val', item_tfms=Resize(224),
                                   batch_tfms=Normalize.from_stats(*imagenet_stats), bs=32)
model_prepared, qmodel = Quantizer().quantize(model, dls)
export_model_sq8(qmodel, model_prepared, "md/model.bin")
