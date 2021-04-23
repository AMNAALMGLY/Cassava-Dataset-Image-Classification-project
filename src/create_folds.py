from sklearn.model_selection import StratifiedKFold
import pandas as pd
def run():
  train_path = './data/train.csv'
  df = pd.read_csv(train_path)
  df = df.sample(frac=1).reset_index(drop=True)
  df['kfold'] = -1
  y  = df['class'].values

  # initiate the kfold class from model_selection module
  kf =StratifiedKFold(n_splits=5)
  # fill the new kfold column
  for f, (t_, v_) in enumerate(kf.split(X=df, y=y)): 
    df.loc[v_, 'kfold'] = f
  
  df.to_csv('./data/train_Kfold.csv')
