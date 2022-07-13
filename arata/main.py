import torch
from torch import NoneType
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


# train and save original network
def main(is_compress):
    '''torch.set_num_threads(8)
    print(torch.__config__.parallel_info()) 
    sys.exit()'''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    compressed_size = [101, 301, 101]

    if is_compress:
        PATH = '../original/data/original_model.pth'
        original_model = NeuralNetwork().to(device)
        original_model.load_state_dict(torch.load(PATH))

        extract_loader = extract_dataloader()
        feature_dict = feature_extractor(original_model, extract_loader, device)
        PATH = 'data/compressed_model.pth'
        compressed_model = compress(original_model, compressed_size, feature_dict, extract_loader)
        torch.save(compressed_model.state_dict(), PATH)
    test(device)

    '''if is_load:
        test(model_type, device)
        return
    else:
        if model_type == 'shallow':
            compressed_size = [101]
            PATH = '../original/data/original_model.pth'
            original_model = ShallowNeuralNetwork().to(device)

            original_model.load_state_dict(torch.load(PATH))

            extract_loader = extract_dataloader()
            feature_dict = feature_extractor(original_model, extract_loader, device)
            PATH = 'data/compressed_shallow_model.pth'
        elif model_type == 'normal':
            compressed_size = [11, 11, 11]
            PATH = '../original/data/original_shallow_model.pth'
            original_model = NeuralNetwork().to(device)

            original_model.load_state_dict(torch.load(PATH))

            extract_loader = extract_dataloader()
            feature_dict = feature_extractor(original_model, extract_loader, device)
            PATH = 'data/compressed_model.pth'
        compressed_model = compress(original_model, compressed_size, feature_dict, extract_loader)
        torch.save(compressed_model.state_dict(), PATH)'''

# Test the saved model.
def test(device): 
    compressed_size = []
    PATH = 'data/compressed_model.pth'
    tens = torch.load(PATH)
    for idx, key in enumerate(tens.keys()):
        if idx%2 == 1:
            continue
        compressed_size.append(tens[key].size()[0])
    compressed_size.pop(-1)
    print(compressed_size)

    compressed_model = LoadNeuralNetwork(compressed_size).to(device)
    PATH = 'data/compressed_model.pth'
    compressed_model.load_state_dict(torch.load(PATH))

    train_dataloader, test_dataloader = dataloader(10000)
    loss_fn = nn.CrossEntropyLoss()
    print('test error')
    test_loop(test_dataloader, compressed_model, loss_fn, device)
    print('train error')
    test_loop(train_dataloader, compressed_model, loss_fn, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--compress', action='store_true', help='compress from scratch?')
    args = parser.parse_args() 
    is_compress = args.compress
    main(is_compress)


    '''original_model = NeuralNetwork().to(device)
    PATH = '../original/data/original_model.pth'
    original_model.load_state_dict(torch.load(PATH))

    extract_loader = extract_dataloader()
    feature_dict = feature_extractor(original_model, extract_loader, device)

    EVD(feature_dict, 0)'''

    '''tmp = extract_loader.__iter__()
    input, target = tmp.next()
    print(input.size())
    input = torch.flatten(input, 1, -1)
    print(input.size())

    input = feature_dict['layer_1']
    print(input.size())
    sys.exit()'''

    '''x = torch.tensor([0, 1, 2, 3, 4])
    y = torch.tensor([0, 1, 2, 3, 4])
    c = {'c': [x, y]}
    torch.save(c, 'tensor.pt')
    sys.exit()
    
    x = original_model.linear_relu_stack[0].bias
    print(original_model.linear_relu_stack[0].bias.size())
    y = torch.t(torch.unsqueeze(x, 0))
    print(y.size())
    z = torch.squeeze(x)
    print(z.size())
    sys.exit()
    '''