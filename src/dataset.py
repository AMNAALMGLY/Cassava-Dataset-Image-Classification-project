from torch.utils.data  import Dataset
from PIL               import Image
import pandas as pd
from src import configs
class CassavaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataFrame, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cassavaframe =dataFrame
        self.transform = transform
        self.class_to_indice={'cmd':0,'cbb':1,'cbsd':2,'healthy':3,'cgm':4, 'no_class':-1}

    def __len__(self):
        return len(self.cassavaframe)

    def __getitem__(self, idx):
        fileName = self.cassavaframe.loc[idx,'path']
        classCategory = self.cassavaframe.loc[idx,'class']
        im = Image.open(fileName)
        if self.transform:
            im = self.transform(im)
            
        return im.view(3,configs.parameters['image size'],configs.parameters['image size']), self.class_to_indice[classCategory], fileName
