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

from src.dataset import CassavaDataset
import argparse
import random
import numpy as np
import os
import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
models=os.listdir(configs.SAVE_MODEL)

model_class=torchvision.models.resnext50_32x4d(pretrained=True)
model_class.fc=nn.Linear(2048,5)
df = pd.read_csv(configs.TEST_PATH)


test_transforms = transforms.transforms['test_t1']
val_data   = CassavaDataset(df,transform=test_transforms)

batch_size  = configs.parameters['batch size']
test_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,)

pred=pd.DataFrame(0,columns=['Catogary','Id' ,'0', '1', '2', '3', '4'],index=range(0,len(val_data)))

classes_1 = ['0', '1', '2', '3', '4']

index_to_class={0:'cmd',1:'cbb',2:'cbsd',3:'healthy',4:'cgm',}

for model in models[1:3]:
  model=os.path.join(configs.SAVE_MODEL,model)
  model_class.load_state_dict(torch.load(model))
  model_class=model_class.to(device)
  model_class.eval()
  
  with torch.no_grad():
    i=0
    #print(test_loader.dataset.class_to_indice)
    for batch,_,names in test_loader:
            #print(i)
            #print(batch)
            file_names=[name.split('/')[-1] for name in names]
            
            #print("before pred")
            r = model_class(batch.to(device))
            #print("r",r)
         
            pred.loc[range(i*batch_size,min((i+1)*batch_size,len(val_data))),classes_1]+=r.cpu().numpy()
            pred.loc[range(i*batch_size,min((i+1)*batch_size,len(val_data))),'Id']=file_names
            #pred.loc[range(i*batchsize,min((i+1)*batchsize,len(test_data))),'model']=model
            i+=1

pred[classes_1]=pred[classes_1]/len(models[:3])
pred['Catogary']=pred.loc[:,classes_1].idxmax(axis=1).astype('int')
pred['Catogary']=pred['Catogary'].apply(lambda x:index_to_class[x])
pred.drop(columns=classes_1,inplace=True)
pred.to_csv(os.path.join(configs.SUB_PATH,'submission.csv'))
#pred['Catogary']=torch.