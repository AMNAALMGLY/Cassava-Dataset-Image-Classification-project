import torchvision
import torch.nn as nn
def get_model(model_ID):
  model = None
  if model_ID == 'resnext_50':
    resnext_model = torchvision.models.resnext50_32x4d(pretrained=True)
    resnext_model.fc =nn.Linear(2048,5)
    model = resnext_model
  elif model_ID == 'resnet_50':
    resnet_model = torchvision.models.resnet50(pretrained=True)
    resnet_model.fc =nn.Linear(2048,5)
    model = resnet_model
  return model

