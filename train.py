# Don't edit this file! This was automatically generated from "train.ipynb".

from fastai.vision.all import *
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#path = untar_data(URLs.IMAGENETTE) # this downloads to ~/.fastai/data/imagenette2
models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
}
def train(model_name, epochs=1):
    if model_name not in models:
        raise ValueError(f"Model name '{model_name}' is not supported. Choose from: {list(models.keys())}")
 
    model = models[model_name]

    path = Path('data/imagenette2')
    # ensure that transformed data is not used as well
    train_fnames = get_image_files(path/'train')
    val_fnames = get_image_files(path/'val')
    fnames = train_fnames + val_fnames

    # load the data
    dls = ImageDataLoaders.from_path_func(
        path, fnames, label_func=parent_label, item_tfms=Resize(224))
    
    # setup model with pretrained weights and data
    learn = vision_learner(dls, model, metrics=CrossEntropyLossFlat(), 
                           pretrained=True)

    # train with model frozen except for the last layer
    learn.fine_tune(epochs)

    # save the model to models folder
    models_dir = Path('models')
    # create the models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f'{model_name}.pt'
    torch.save(learn.model, save_path)

def load(model_name):
    model = torch.load(f'models/{model_name}.pt', map_location=device)
    model.eval()
    return model
