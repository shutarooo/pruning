import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
#import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import *
from parameters import *
from spectral_pruning import *

import sys
sys.path.append('../')

from original.loop import *
from original.model import *
from original.dataloader import *

import sys



def main():

    '''device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    original_model = ShallowNeuralNetwork().to(device)
    PATH = '../original/data/original_shallow_model.pth'
    original_model.load_state_dict(torch.load(PATH))

    extract_loader = extract_dataloader()
    feature_dict = shallow_feature_extractor(original_model, extract_loader, device)
    compressed_model, size = compress(original_model, feature_dict, device)
    PATH = 'data/compressed_shallow_model.pth'
    torch.save(compressed_model.state_dict(), PATH)

    train_dataloader, test_dataloader = dataloader(10000)
    loss_fn = nn.CrossEntropyLoss()
    test_loop(train_dataloader, compressed_model, loss_fn, device)'''
    torch.set_num_threads(8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print('Using {} device'.format(device))

    original_model = NeuralNetwork().to(device)
    PATH = '../original/data/original_model.pth'
    original_model.load_state_dict(torch.load(PATH))

    extract_loader = extract_dataloader()
    feature_dict = feature_extractor(original_model, extract_loader, device)
    compressed_model, size = compress(original_model, feature_dict, device)
    PATH = 'data/compressed_model.pth'
    torch.save(compressed_model.state_dict(), PATH)

    train_dataloader, test_dataloader = dataloader(10000)
    loss_fn = nn.CrossEntropyLoss()
    test_loop(train_dataloader, compressed_model, loss_fn, device)


if __name__ == "__main__":
    main()