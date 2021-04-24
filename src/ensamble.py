import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data  import Dataset
from torch.autograd    import Variable
from torch.optim       import lr_scheduler
from torch.utils.data  import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision       import transforms, datasets, models
from os                import listdir, makedirs, getcwd, remove
from os.path           import isfile, join, abspath, exists, isdir, expanduser
from PIL               import Image
import pandas as pd
from src import configs
from src import model_dispatcher
from src import create_schedular
from src import transforms
from src import models
from src import data
from src.dataset import CassavaDataset
import argparse
import random
import numpy as np
import os
import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
models=os.listdir('./models')
model_class=torchvision.models.resnext50_32x4d(pretrained=True)
df = pd.read_csv(configs.TEST_PATH)


test_transforms = transforms.transforms['test_t1']
val_data   = CassavaDataset(df,transform=test_transforms)

batch_size  = configs.parameters['batch size']
test_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,)

pred=pd.DataFrame(0,columns=['Catogary','Id'],index=range(0,len(val_data)))
for model in models[:3]:
  model_class.load_state_dict(torch.load(model))
  model.eval()
  
  with torch.no_grad():
    i=0
    for batch,_ ,names in test_loader:
            
            file_names=[name.split('/')[-1] for name in names]
            
          
            r = model(batch.to(device))
           
         
            pred.loc[range(i*batchsize,min((i+1)*batchsize,len(test_data))),'Catogary']+=r.m(dim=1).cpu()
            pred.loc[range(i*batchsize,min((i+1)*batchsize,len(test_data))),'Id']=file_names
            #pred.loc[range(i*batchsize,min((i+1)*batchsize,len(test_data))),'model']=model
            i+=1

pred['Catogary']=pred/len(models[:3])