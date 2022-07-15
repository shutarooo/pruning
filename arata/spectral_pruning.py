from curses import A_DIM
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
#import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import math
import sys
import json

import sys
sys.path.append('../')

from suzuki.model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

layer_num = 3
original_size = [300, 1000, 300, 10]
layer_keys = ['layer_1', 'layer_2', 'layer_3', 'out']
n = 60000

layer_num = 1
original_size = [1000, 10]
layer_keys = ['layer_1', 'out']
n = 60000

def EVD(feature_dict, layer_idx):
    layer = layer_keys[layer_idx]
    output = feature_dict[layer]
    cov_mx =  torch.t(output) @ output
    #print(torch.min(cov_mx))
    e_values, e_vectors = torch.linalg.eig(cov_mx)
    e_vectors = output @ e_vectors.to(torch.float)
    '''for i in range(len(e_values)):
        if e_values[i] < e_values[i+1]:
            print('----------------')'''

    '''print(len(e_vectors[:, 0]))
    for i in range(30):
        cnt = 0
        for v in e_vectors[:, i]:
            if v>= 0:
                cnt += 1
        print('the number of positive elements in e_vector[{}] is {}'.format(i,cnt))
    '''
    #print('e_values'e_values[:5])
    #print(e_vectors[:,0])
    return e_values, e_vectors

def construct(W_list, A_list, feature_dict, original_model, extract_loader, compressed_size):
    
    weight_list = []
    bias_list = []
    ReLU = torch.nn.ReLU()
    for layer_idx in range(layer_num+1):
        print("------- {} layer --------".format(layer_idx))
        # For the 1st layer, only set compressed weight.
        if layer_idx==0:
            weight_list.append(W_list[layer_idx])     #
            continue

        # Calculate reshape matrix.
        At = A_list[layer_idx-1]             #
        A = torch.t(A_list[layer_idx-1])     #
        P = torch.t(feature_dict[layer_keys[layer_idx-1]]) @ A @ torch.inverse(At@A)
        print('P: {}'.format(P.size()))

        # Estimate bias.
        input = None
        if layer_idx == 1:
            tmp = extract_loader.__iter__()
            input, target = tmp.next()
            input = torch.flatten(input, 1, -1)
        else:
            input = feature_dict[layer_keys[layer_idx-2]]
        apx = P @ ReLU(W_list[layer_idx-1] @ torch.t(input))      #

        error = torch.t(feature_dict[layer_keys[layer_idx-1]]) - apx
        b_mean = torch.mean((error-apx), 1, True)
        b_opt = torch.linalg.lstsq(P, b_mean).solution
        b_opt = torch.squeeze(b_opt)
        bias_list.append(b_opt/100000)
        #print('b_mean: {}'.format(b_opt.size()))
        #sys.exit()

        # Calculate complete weight for compressed model on this layer.
        W = original_model.linear_relu_stack[layer_idx*2].weight if layer_idx==layer_num else W_list[layer_idx]   #
        print('W: {}'.format(W.size()))
        weight_list.append(W @ P)
    bias_original = original_model.linear_relu_stack[layer_num*2].bias
    bias_list.append(bias_original)
    compressed_model = CompressedShallowNeuralNetwork(weight_list, bias_list, compressed_size)
    return compressed_model
    


def compress(original_model, compressed_size, feature_dict, extract_loader):
    
    ReLU = torch.nn.ReLU()
    W_list = []
    A_list = []
    for layer_idx in range(layer_num):
        print("------- {} layer --------".format(layer_idx))
        e_values, e_vectors = EVD(feature_dict, layer_idx)
        A = None
        W = None
        for j in range(int((compressed_size[layer_idx]+1)/2)):
            e_vector = e_vectors[:,j] / torch.linalg.norm(e_vectors[:,j])

            input = None
            if layer_idx == 0:
                tmp = extract_loader.__iter__()
                input, target = tmp.next()
                input = torch.flatten(input, 1, -1).to(device)
            else:
                input = feature_dict[layer_keys[layer_idx-1]]

            #print(input.device)
            #print(e_vector.device) 
            W_plus_info = torch.linalg.lstsq(input, torch.t(e_vector).to(device))
            W_plus = W_plus_info.solution
            W_minus = -1 * W_plus

            e_vector_plus = ReLU(e_vector)
            e_vector_minus = ReLU(-1 * e_vector)
            if A == None:
                if torch.max(e_vector_plus) <= 0:
                    A = torch.unsqueeze(e_vector_minus, 0)
                else:
                    A = torch.unsqueeze(e_vector_plus, 0)
            else:
                A = torch.cat((A, torch.unsqueeze(e_vector_plus, 0)), 0)
                A = torch.cat((A, torch.unsqueeze(e_vector_minus, 0)), 0)

            if W == None:
                if torch.max(e_vector_plus) <= 0:
                    W = torch.unsqueeze(W_minus, 0)
                else:
                    W = torch.unsqueeze(W_plus, 0)
            else:
                W = torch.cat((W, torch.unsqueeze(W_plus, 0)), 0)
                W = torch.cat((W, torch.unsqueeze(W_minus, 0)), 0)
            print(A.size())
            print(W.size())

            print(j, W_plus.size(), W_plus_info.residuals)
        W_list.append(W)
        A_list.append(A)

    '''d = {'W_list': W_list, 'A_list': A_list}
    torch.save(d, 'data/WA.pt')
    #sys.exit()
    


    WA = torch.load('data/WA.pt')
    W_list = WA['W_list']
    A_list = WA['A_list']'''
    compressed_model = construct(W_list, A_list, feature_dict, original_model, extract_loader, compressed_size)
    return compressed_model


