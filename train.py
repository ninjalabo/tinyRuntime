# Don't edit this file! This was automatically generated from "train.ipynb".

from fastai.vision.all import *
from huggingface_hub import from_pretrained_fastai, push_to_hub_fastai
repo ='jkokko/'

models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
}

def train(model_name, epochs=1):
    if model_name not in models:
        raise ValueError(f"Model name '{model_name}' is not supported. Choose from: {list(models.keys())}")
 
    model = models[model_name]
    path = untar_data(URLs.IMAGENETTE_320,data=Path.cwd()/'data')
    dls = ImageDataLoaders.from_folder(path, valid='val', item_tfms=Resize(224), batch_tfms=Normalize.from_stats(*imagenet_stats),)
    learn = vision_learner(dls, model, metrics=accuracy, pretrained=True)
    learn.fine_tune(epochs)

    return learn

def train_all(epochs=1):
    for model_name in models:
        learn = train(model_name, epochs)
        push_to_hub_fastai(learner=learn, repo_id=repo + model_name)
        


def load(model_name):
    learn = from_pretrained_fastai(repo_id=repo + model_name)
    return learn
