import torch
from torch.utils.data import Dataset
import numpy as np
from .LaprasDataProcessing import laprasLoader
from .CasasDataProcessing import casasLoader
from .OpportunityDataProcessing import opportunityLoader
from .ArasDataProcessing import arasLoader
from .OpenPackDataProcessing import openpackLoader
from .PAMAPDataProcessing import pamapLoader

import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from .augmentations import PERMUTE, select_transformation

from tsaug import *

# for storing dataset element
class TSDataSet:
    def __init__(self,data, label, length, user_id):
        self.data = data
        self.label = int(label)
        self.length= int(length)
        self.user_id = int(user_id)      

# use for lapras dataset
def label_num(filename):
    label_cadidate = ['Chatting', 'Discussion', 'GroupStudy', 'Presentation', 'NULL']
    label_num = 0
    for i in range(len(label_cadidate)):
        if filename.find(label_cadidate[i]) > 0:
            label_num = i+1    
    return label_num

# use for dataset normalization 
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return round(df_norm,3)

def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std

class TimeseriesDataset(Dataset):   
    def __init__(self, data, window, target_cols):
        self.data = torch.Tensor(data)
        self.window = window
        self.target_cols = target_cols
        self.shape = self.__getshape__()
        self.size = self.__getsize__() 
    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        y = self.data[index+self.window,0:target_cols]
        return x, y 
    def __len__(self):
        return len(self.data) -  self.window     
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)    
    def __getsize__(self):
        return (self.__len__())



