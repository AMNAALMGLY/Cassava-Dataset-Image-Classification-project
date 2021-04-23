import torch
from torchvision       import transforms, datasets, models
from src import configs
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transforms = {
  'train_t1':transforms.Compose([transforms.RandomRotation(configs.parameters['rotation']),
      transforms.RandomResizedCrop(configs.parameters['image size']),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)]),
  'test_t1':transforms.Compose([
      transforms.CenterCrop(configs.parameters['image size']),                             transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])

}



