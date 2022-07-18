import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
#import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spectral_pruning import *

import sys
sys.path.append('../')

from original.loop import *
from original.model import *
from original.dataloader import *

import sys
import argparse

X = torch.tensor([
    [1, 1, -4, 8, -3, -2],
    [2, 2, 2, 1, -3, -1], 
    [1, 1, 7, 1, -4, -1], 
    [2, 2, 6, 1, -8, -1], 
    [3, 3, 2, 1, -2, -8], 
    [1, 1, 7, 2, -9, 2],
    ]).float()

a = torch.t(torch.tensor([[1,3,5,2,-1,4]])).float()

X = torch.tensor([
    [1, 1, -4],
    [2, 2, 2], 
    [1, 1, 7], 

    ]).float()

a = torch.t(torch.tensor([[-2,6,9]])).float()

gaussian_elimination(X,a)

'''x = torch.tensor([[1 for i in range(5)]])
X = torch.tensor([
    [1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2], 
    [4, 4, 4, 4, 4, 4], 
    [8, 8, 8, 8, 8, 8]
    ]).float()
X = torch.t(X)'''