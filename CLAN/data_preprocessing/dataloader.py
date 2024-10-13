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

def visualization_data(dataset_list, file_name, activity_num):
    print("Visualizing Dataset --------------------------------------")
    label_count = [0 for x in range(activity_num)]
    # for visualization
    for k in range(len(dataset_list)):
        visual_df = pd.DataFrame(dataset_list[k].data)

        fig, ax = plt.subplots(figsize=(10, 6))
        axb = ax.twinx()

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)

        # Plotting on the first y-axis
        for i in range(len(dataset_list[0].data[0])):
            ax.plot(visual_df[i], label = str(i+1))

        ax.legend(loc='upper left')
        
        plt.savefig(file_name+'visualization/'+str(dataset_list[k].label)+'_'+str(label_count[dataset_list[k].label-1])+'.png')
        plt.close(fig)
        label_count[dataset_list[k].label-1]+=1

    print("Visualizing Dataset Finished--------------------------------------")



# A method finds types of labels and counts the number of each label
def count_label(dataset_list):
    # find types and counts of labels
    types_label_list = []
    count_label_list = []

    for i in range(len(dataset_list)):
        if(dataset_list[i].label not in types_label_list):
            types_label_list.append(dataset_list[i].label)
            count_label_list.append(1)
        else:
            count_label_list[types_label_list.index(dataset_list[i].label)]+=1

    print('types_label :', types_label_list)
    print('count_label :', count_label_list) 
    print('sum of # episodes:', sum(count_label_list))  
                
    return types_label_list, count_label_list


def count_label_labellist(labellist):
    # finding types and counts of label
    types_label_list =[]
    count_label_list = []
    for i in range(len(labellist)):
        if(labellist[i] not in types_label_list):
            types_label_list.append(labellist[i])
            count_label_list.append(1)
        else:
            count_label_list[types_label_list.index(labellist[i])]+=1

    print('types_label :', types_label_list)
    print('count_label :', count_label_list)   
    print('sum of # episodes:', sum(count_label_list))
                
    return types_label_list, count_label_list

