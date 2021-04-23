import torch

def get_schedular(ID, optimizer):
  if ID == 'constant':
    return None
  elif ID == 'cosine_1':
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7,verbose=True) 