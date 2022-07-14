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
import argparse



def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    print('is_comress: {}'.format(is_compress))

    if is_compress:
        original_model = ShallowNeuralNetwork().to(device)
        PATH = '../original/data/original_shallow_model.pth'
        original_model.load_state_dict(torch.load(PATH))

        extract_loader = extract_dataloader()
        feature_dict = shallow_feature_extractor(original_model, extract_loader, device)
        compressed_model, size = compress(original_model, feature_dict, device)
        PATH = 'data/compressed_shallow_model.pth'
        torch.save(compressed_model.state_dict(), PATH)
    test(device)

# Test the saved model.
def test(device): 
    compressed_size = []
    weight_list = []
    bias_list = []
    PATH = 'data/compressed_shallow_model.pth'
    tens = torch.load(PATH)
    for idx, key in enumerate(tens.keys()):
        if idx%2 == 1:
            bias_list.append(tens[key])
            continue
        weight_list.append(tens[key])
        compressed_size.append(tens[key].size()[0])
    compressed_size.pop(-1)
    print(compressed_size)

    compressed_model = CompressedShallowNeuralNetwork(weight_list, bias_list, compressed_size).to(device)
    '''PATH = 'data/compressed_shallow_model.pth'
    compressed_model.load_state_dict(torch.load(PATH))'''

    train_dataloader, test_dataloader = dataloader(10000)
    loss_fn = nn.CrossEntropyLoss()
    print('test error')
    test_loop(test_dataloader, compressed_model, loss_fn, device)
    print('train error')
    test_loop(train_dataloader, compressed_model, loss_fn, device)

#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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--compress', action='store_true', help='compress from scratch? True or False')
    args = parser.parse_args() 
    is_compress = args.compress
    main()