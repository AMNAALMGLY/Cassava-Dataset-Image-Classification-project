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

MODEL_ID = ''
FOLD_ID =''
SCHEDULER_ID=''

def run(fold, model, scheduler):
  global MODEL_ID,SCHEDULER_ID,FOLD_ID
  MODEL_ID=model
  SCHEDULER_ID=scheduler
  FOLD_ID = 'fold' + str(fold) + '_'

  # set the seed for reproducaility
  set_seed(2021)
  # read the training data with folds
  df = pd.read_csv(configs.KFOLD_CSV)
  # training data is where kfold is not equal to provided fold # also, note that we reset the index
  df_train = df[df.kfold != fold].reset_index(drop=True)
  # validation data is where kfold is equal to provided fold
  df_valid = df[df.kfold == fold].reset_index(drop=True)
  # drop the label column from dataframe and convert it to # a numpy array by using .values.
  # target is label column in the dataframe


  #============================ Transforms and Data ==============================#
  train_transforms = transforms.transforms['train_t1']
  test_transforms = transforms.transforms['test_t1']

  

  val_data   = CassavaDataset(df_valid,transform=test_transforms)
  train_data = CassavaDataset(df_train, transform=train_transforms)


  batch_size  = configs.parameters['batch size']
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,)
  valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                              )
  # ============================= Model ==========================================#
  model = model_dispatcher.get_model(model)

 
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=configs.parameters['lr'])
  scheduler_glob = create_schedular.get_schedular(scheduler, optimizer)
  
  #============================== fit ============================================#
  num_epochs = configs.parameters['epochs']
  besmodel, losses = train(model,criterion, train_loader, optimizer, num_epochs=num_epochs,scheduler=scheduler_glob,valid_loader=valid_loader)
  
  

def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if torch.cuda.is_available():
    print('Gpu is available')
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
  print('end')

def train(model, criterion, data_loader, optimizer, num_epochs,scheduler,valid_loader=None):
  """Simple training loop for a PyTorch model.""" 
  
  path_save_model= configs.SAVE_MODEL
  best_model = None
  best_accu = 0
  model.to(device)

  ema_loss = None

  print('----- Training Loop -----')
  f_losses = []
  for epoch in range(num_epochs):
      model.train()
      losses = []
      for batch_idx, (features, target,_) in enumerate(data_loader):
        
          output = model(features.to(device))
          
          loss = criterion(output.to(device), target.to(device))
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          if ema_loss is None:
              ema_loss = loss.item()
          else:
              ema_loss += (loss.item() - ema_loss) * 0.01
    
      f_losses.append(ema_loss)
      
      scheduler.step()
      print('Epoch: {} \tLoss: {:.6f}'.format(epoch, ema_loss),)
         
      
      current_accu = test(model, valid_loader,32,verbose=False)
      print("current accu {}".format(current_accu))
      if current_accu > best_accu:
          best_accu = current_accu
          model_path = os.path.join(path_save_model, FOLD_ID + MODEL_ID + '_' +SCHEDULER_ID + '_'+ str(best_accu)[0:4] + '.pt') 
          torch.save(model.state_dict(), model_path)
          best_model = model

      print("best accu is {}%".format(best_accu))
    
  return best_model, f_losses

def test(model, data_loader, batch_size,verbose=True):
  """Measures the accuracy of a model on a data set.""" 
  # Make sure the model is in evaluation mode.
  model.eval()
  correct = 0
  if verbose:
      print('----- Model Evaluation -----')
  # We do not need to maintain intermediate activations while testing.
  with torch.no_grad():
      # Loop over test data.
      for features, target, _ in data_loader:
          # Forward pass.
          output = model(features.to(device))
          
          # Get the label corresponding to the highest predicted probability.
          pred = output.argmax(dim=1, keepdim=True)
          
          # Count number of correct predictions.
          correct += pred.cpu().eq(target.view_as(pred)).sum().item()

  # Print test accuracy.
  percent = 100. * correct / (len(data_loader.sampler))
  if verbose:
      print(f'Test accuracy: {correct} / {(len(data_loader.sampler))} ({percent:.0f}%)')
  #torch.save(model.state_dict(), 'model.ckpt')
  return percent



if __name__ == '__main__':
  # initialize ArgumentParser class of argparse parser = argparse.ArgumentParser()
  # add the different arguments you need and their type # currently, we only need fold
  parser = argparse.ArgumentParser()

  parser.add_argument(
  "--fold", type=int
  )
  
  parser.add_argument(
  "--model", type=str
  )
  
  parser.add_argument(
  "--scheduler", type=str
  )
  args = parser.parse_args()

  run(
    args.fold,
    args.model,
    args.scheduler
  )