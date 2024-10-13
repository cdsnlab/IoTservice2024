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

def padding_by_max(lengthlist, normalized_df):
   
    # reconstruction of datalist    
    datalist=[]
    reconst_list =[]
    count_lengthlist = 0
    print("max padding (length): ", max(lengthlist))

    # reconstruction of normalized list    
    # for each row
    for i in range(len(lengthlist)):
        reconst_list =[]    
        # cut df by each length
        for j in range(count_lengthlist,(count_lengthlist+lengthlist[i])):
            reconst_list.append(normalized_df.iloc[j,:].tolist())            
        count_lengthlist += lengthlist[i]

        #padding to each data list
        if((max(lengthlist)-lengthlist[i])%2 == 0):
            p2d = (0, 0, int((max(lengthlist)-lengthlist[i])/2), int((max(lengthlist)-lengthlist[i])/2))
        else :
            p2d = (0, 0, int((max(lengthlist)-lengthlist[i]+1)/2)-1, int((max(lengthlist)-lengthlist[i]+1)/2))
        datalist.append(F.pad(torch.tensor(reconst_list),p2d,"constant", -1))
        
    # convert to tensor    
    datalist = torch.stack(datalist)
    return datalist

def padding_by_mean(lengthlist, normalized_df):
    # reconstruction of datalist    
    datalist=[]
    reconst_list =[]
    count_lengthlist = 0
    mean_length = int(sum(lengthlist)/len(lengthlist))
    print("mean padding (length):", mean_length)
    for i in range(len(lengthlist)):
        reconst_list =[]    
        # cut df by each length
        if(lengthlist[i]>= mean_length): # length is larger than mean
            for j in range(count_lengthlist, count_lengthlist+mean_length):
                reconst_list.append(normalized_df.iloc[j,:].tolist())
            datalist.append(torch.tensor(reconst_list))
        else: # length is smaller than mean
            for j in range(count_lengthlist, (count_lengthlist+lengthlist[i])):
                reconst_list.append(normalized_df.iloc[j,:].tolist())
            # padding to the end 
            p2d = (0, 0, 0, mean_length-lengthlist[i])
            datalist.append(F.pad(torch.tensor(reconst_list),p2d,"constant", 0))    
        count_lengthlist += lengthlist[i]
    
    # convert to tensor    
    datalist = torch.stack(datalist)    
    return datalist

def reconstrct_list(length_list, normalized_df):
    
    # reconstruction of datalist    
    data_list=[]
    reconst_list =[]
    count_lengthlist = 0

    # for each row
    for i in range(len(length_list)):
        reconst_list =[]    
        # append by each length
        for j in range(count_lengthlist,(count_lengthlist+length_list[i])):
            reconst_list.append(normalized_df.iloc[j,:].tolist())            
        count_lengthlist += length_list[i]
        data_list.append(torch.tensor(reconst_list))
    return data_list

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

def padding_by_max(lengthlist, normalized_df):
   
    # reconstruction of datalist    
    datalist=[]
    reconst_list =[]
    count_lengthlist = 0
    print("max padding (length): ", max(lengthlist))

    # reconstruction of normalized list    
    # for each row
    for i in range(len(lengthlist)):
        reconst_list =[]    
        # cut df by each length
        for j in range(count_lengthlist,(count_lengthlist+lengthlist[i])):
            reconst_list.append(normalized_df.iloc[j,:].tolist())            
        count_lengthlist += lengthlist[i]

        #padding to each data list
        if((max(lengthlist)-lengthlist[i])%2 == 0):
            p2d = (0, 0, int((max(lengthlist)-lengthlist[i])/2), int((max(lengthlist)-lengthlist[i])/2))
        else :
            p2d = (0, 0, int((max(lengthlist)-lengthlist[i]+1)/2)-1, int((max(lengthlist)-lengthlist[i]+1)/2))
        datalist.append(F.pad(torch.tensor(reconst_list),p2d,"constant", -1))
        
    # convert to tensor    
    datalist = torch.stack(datalist)
    return datalist

def padding_by_mean(lengthlist, normalized_df):
    # reconstruction of datalist    
    datalist=[]
    reconst_list =[]
    count_lengthlist = 0
    mean_length = int(sum(lengthlist)/len(lengthlist))
    print("mean padding (length):", mean_length)
    for i in range(len(lengthlist)):
        reconst_list =[]    
        # cut df by each length
        if(lengthlist[i]>= mean_length): # length is larger than mean
            for j in range(count_lengthlist, count_lengthlist+mean_length):
                reconst_list.append(normalized_df.iloc[j,:].tolist())
            datalist.append(torch.tensor(reconst_list))
        else: # length is smaller than mean
            for j in range(count_lengthlist, (count_lengthlist+lengthlist[i])):
                reconst_list.append(normalized_df.iloc[j,:].tolist())
            # padding to the end 
            p2d = (0, 0, 0, mean_length-lengthlist[i])
            datalist.append(F.pad(torch.tensor(reconst_list),p2d,"constant", 0))    
        count_lengthlist += lengthlist[i]
    
    # convert to tensor    
    datalist = torch.stack(datalist)    
    return datalist

def reconstrct_list(length_list, normalized_df):
    
    # reconstruction of datalist    
    data_list=[]
    reconst_list =[]
    count_lengthlist = 0

    # for each row
    for i in range(len(length_list)):
        reconst_list =[]    
        # append by each length
        for j in range(count_lengthlist,(count_lengthlist+length_list[i])):
            reconst_list.append(normalized_df.iloc[j,:].tolist())            
        count_lengthlist += length_list[i]
        data_list.append(torch.tensor(reconst_list))
    return data_list

def data_augmentation(dataset_list, aug_method, aug_wise):
    # Data Augmentation Module
    print('-' * 100)
    print(('*'*5)+'Augmentation Starting'+('*'*5))
    
    # For give the same number of data size (balancing the numbers)
    types_label_list, count_label_list = count_label(dataset_list)
    max_label_count = max(count_label_list)

    # calculating the numbers that need to be augmented
    sub_count_label = [0] * len(types_label_list)
    for i in range(len(types_label_list)):
        sub_count_label[i] = max_label_count - count_label_list[i]
    print("The amount of augmented data:", sub_count_label)
    
    copy_count_label = sub_count_label.copy()

# temporal aspect data augmentation
    if aug_wise == 'Temporal' or aug_wise == 'Temporal2' :     
        
        for i in range(len(dataset_list)): 
            # Augmentation for data balancing
            target_label = types_label_list.index(dataset_list[i].label)
            target_data  = dataset_list[i].data

            # Target data shape : (N, T, C)
            
            for j in range(math.ceil(sub_count_label[target_label]/count_label_list[target_label])): 
            #print(dataset_list[i].label, "" , math.ceil(sub_count_label[types_label_list.index(dataset_list[i].label)]/count_label_list[type   s_label_list.index(dataset_list[i].label)]))
                if copy_count_label[target_label] > 0:
                # print(copy_count_label[types_label_list.index(dataset_list[i].label)],"and",sub_count_label[types_label_list.index(dataset_list[i].label)])          
                    #print("Aug", dataset_list[i].data.shape)
                    # select data transformation
                    trans = select_transformation('AddNoise', target_data.shape[0])
                    #trans = PERMUTE(min_segments=1, max_segments=5, seg_mode="random")
                    aug = trans.augment(np.reshape(target_data,(1, target_data.shape[0], -1)))
                    #print("Aug_after", aug.shape, aug[0].shape)  
                    ts_ds = TSDataSet(aug[0], dataset_list[i].label, len(aug[0]), dataset_list[i].user_id)
                    dataset_list.append(ts_ds)
                    copy_count_label[target_label] = copy_count_label[target_label]-1   
        
        if aug_wise == 'Temporal2' :
            for i in range(len(dataset_list)): 
                target_data  = dataset_list[i].data
                trans = select_transformation('AddNoise', target_data.shape[0])
                #trans = PERMUTE(min_segments=1, max_segments=3, seg_mode="random")
                aug = trans.augment(np.reshape(target_data,(1, target_data.shape[0], -1))) 
                ts_ds = TSDataSet(aug[0], dataset_list[i].label, len(aug[0]), dataset_list[i].user_id)
                dataset_list.append(ts_ds)

# a method to create continuous labels from 0 due to data process
def sort_data_label(dataset_list):
    # change labels
    types_label_list, _ = count_label(dataset_list)

    types_label_list.sort()
    changed_label_list =[i for i in range(1, len(types_label_list)+1)]

    print("original label:", types_label_list, "\nchanged label:", changed_label_list)    
        
    for i in range(len(dataset_list)): 
        dataset_list[i].label = changed_label_list[types_label_list.index(dataset_list[i].label)]

    return dataset_list 

def sort_only_label(label_list):
    # change labels
    
    # find types of labels
    types_label_list =[]

    for i in range(len(label_list)):
        if(label_list[i] not in types_label_list):
            types_label_list.append(label_list[i])

    types_label_list.sort()
    changed_label_list =[i for i in range(0, len(types_label_list))]

    print("original label:", types_label_list, "\nchanged label:", changed_label_list)    
        
    for i in range(len(label_list)): 
        label_list[i] = changed_label_list[types_label_list.index(label_list[i])]

    return label_list 

def delete_label(data_list, label_list, deleted_label):    
    count = 0
    while count < len(data_list):

        if label_list[count] == (deleted_label-1):
            data_list = torch.cat([data_list[0:count], data_list[count+1:]])
            del label_list[count]
            count = count-1

        count = count+1        
    label_list = sort_only_label(label_list)

    return data_list, label_list

def change_label(label_list, deleted_label):

    for i in range(len(label_list)): 
        if label_list[i] == (deleted_label-1):
            label_list[i] = 100000
    
    sort_only_label(label_list)

    return label_list
