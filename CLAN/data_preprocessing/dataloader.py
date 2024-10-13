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
