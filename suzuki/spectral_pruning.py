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

from parameters import *


from parameters import layer_num, original_size, layer_keys, n
from model import *

def greedy_optimizer(feature_dict, layer_idx, m_sharp, Z, tau_tilde):
    #print('tau_all: {}'.format(len(tau_tilde)))
    
    J = []
    min_loss = 10**8
    m = original_size[layer_idx]
    cov_ff = cov_mx(feature_dict, layer_idx, m, m)
    '''print(torch.linalg.matrix_rank(cov_ff))
    sys.exit()'''
    while len(J) < m_sharp:
        new_neuron = None
        for n in range(original_size[layer_idx]):
            J.append(n)
            if len(J) != len(set(J)):
                J.pop(-1)
                continue
            loss = convex_loss(feature_dict, layer_idx, J, Z, tau_tilde, cov_ff)
            if loss < min_loss:
                min_loss = loss
                new_neuron = n
            J.pop(-1)
        J.append(new_neuron)
        print('J: {}'.format(J))
    return J

def construct(J, tau_all, compressed_size, original_model, feature_dict, device):
    weight_list = []
    bias_list = []
    for i in range(layer_num+1):
        print("------- {} layer --------".format(i))
        m = original_size[i]
        weight_original = original_model.linear_relu_stack[i*2].weight
        bias_original = original_model.linear_relu_stack[i*2].bias
        # for the 1st layer, there isn't reshape matrix A.
        if i==0:
            weight_list.append(weight_original[J[i]])
            bias_list.append(bias_original[J[i]])
            continue
        # calculate A on the previous layer.
        cov_fj = cov_mx(feature_dict, i-1, m, J[i-1])
        cov_jj = cov_mx(feature_dict, i-1, J[i-1], J[i-1])
        tau = []
        #print(J[i-1])
        for j in range(len(tau_all[i-1])):
            if j in J[i-1]:
                tau.append(tau_all[i-1][j])
        tau = torch.tensor(tau).to(device)
        A_hat_minus = cov_fj @ torch.inverse(cov_jj + torch.diag(tau))
        # for the last layer, use original weight and bias.
        if i==layer_num:
            weight_list.append(weight_original @ A_hat_minus)
            bias_list.append(bias_original)
            break
        weight_list.append(weight_original[J[i]] @ A_hat_minus)
        bias_list.append(bias_original[J[i]])
    compressed_model = CompressedNeuralNetwork(weight_list, bias_list, compressed_size)
    return compressed_model

def est_hyper(feature_dict, layer_idx, device):
    print("------- {} layer --------".format(layer_idx))

    # common variable definition 
    m = original_size[layer_idx]
    cov_l = cov_mx(feature_dict, layer_idx, m, m)
    ide = torch.eye(m).to(device)

    # if target is output layer, only return tau
    if layer_idx == layer_num:
        m = original_size[layer_idx]
        tau_tilde = [1/m for i in range(m)]
        return tau_tilde

    # hyper parameter initialization
    lamb = 10**(-1) * torch.trace(cov_l) * 1
    #c = [63, 48, 2210]
    c = [152.69, 434.14, 6480]
    lamb = c[layer_idx]
    #m = [10, 10, 9]
    m_sharp = 100000
    n_hat = None
    # search lamb s.t. m_sharp < m
    while m_sharp > 10:
        
        n_hat = N_hat(layer_idx, lamb, device, feature_dict)
        
        m_sharp = int(5*n_hat * math.log(80*n_hat)) + 1
        print('m_sharp: {}'.format(m_sharp))
        lamb *= 1.2
    lamb /= 1.2
    print('lamb: {}'.format(lamb))

    # from above lamb, calculate tau_tilde
    inv_mx = torch.inverse(cov_l + lamb * ide)
    mx = cov_l @ inv_mx
    tau_tilde = [mx[i][i] / n_hat for i in range(m)]

    return lamb, m_sharp, tau_tilde

# old
def est_hyperw(feature_dict, layer_idx, m, device):
    print("------- {} layer --------".format(layer_idx))
    #n = original_size[layer_idx]

    #//////////////////////////////////////////////////////////////////////////////////////
    
    cov_l = cov_mx(feature_dict, layer_idx, m, m)
    ide = torch.eye(original_size[layer_idx]).to(device)

    #lamb = 10**(-6) * torch.trace(cov_mx(feature_dict, layer_keys[layer_idx], n, n))
    lamb = 10**(-3) * torch.trace(cov_l)
    tau = None
    m_sharp = 100000

    # 1st: fix lamb s.t. m_sharp < m
    while m_sharp > m*0.7:
        inv_mx = torch.inverse(cov_l + lamb * ide)
        n_hat = N_hat(layer_idx, lamb, device, feature_dict)
        print('N_nat: {}'.format(n_hat))
        mx = cov_l @ inv_mx
        tau = [mx[i][i] / n_hat for i in range(m)]
        m_sharp = int(5*n_hat * math.log(80*n_hat)) + 1
        print(m_sharp)
        lamb *= 2
    lamb /= 2
    # 2nd: calculate m_sharp which satisfies the condition
    '''lhs = 0
    for tau_j in tau:
        print(tau_j)
        lhs += 1/tau_j
    m_condition = int((3/5)*lhs/m) + 1

    m_sharp = max(m_sharp, m_condition)'''
    return tau, lamb, m_sharp

def check_condition(m_sharps, tau_tildes, J_sharp):
    for layer_idx in range(layer_num):
        lhs = 0
        for j in range(len(tau_tildes[layer_idx])):
            if j in J_sharp[layer_idx]:
                lhs += 1/tau_tildes[layer_idx][j]
        m = original_size[layer_idx]
        m_sharp = m_sharps[layer_idx]
        condition = lhs <= (5/3)*m*m_sharp
        print('layer: {}, condition is {}'.format(layer_idx, condition))


def compress(original_model, feature_dict, device):

    lambdas = []
    compressed_size = []
    tau_tildes = []
    J_sharp = []
    taus = []

    # At first, fix the hyper parameter for each layers.
    print('@@@@@@@@@@ estimating hyper parameter. @@@@@@@@@@@@@')
    for layer_idx in range(layer_num):
        lamb, m_sharp, tau_tilde = est_hyper(feature_dict, layer_idx, device)
        lambdas.append(lamb)
        compressed_size.append(m_sharp)
        tau_tildes.append(tau_tilde)
    
    # Only for tau_tilde, we need the info for output layer.
    tau_tilde = est_hyper(feature_dict, layer_num, device)
    tau_tildes.append(tau_tilde)

    
    # Next, compress the model by the backward procedure.
    print('@@@@@@@@@@ compressing the original model. @@@@@@@@@@@@')
    J_plus = [i for i in range(10)]
    model = original_model
    W = model.linear_relu_stack[6].weight
    for layer_idx in reversed(range(layer_num)):
        #lamb, m_sharp, tau_tilde_plus = est_hyper(feature_dict, layer_idx, device)
        lamb = lambdas[layer_idx]
        m_sharp = compressed_size[layer_idx]
        tau_tilde = tau_tildes[layer_idx]
        tau_tilde_plus = tau_tildes[layer_idx+1]
        q = Q(tau_tilde_plus, layer_idx, J_plus)
        Z = z(original_size[layer_idx], q, J_plus, W, device)
        #print(m_sharp)
        #print(lamb)
        #print(tau_tilde)
        tau = m_sharp * lamb * torch.tensor(tau_tilde).to(device)
        #print(tau.size())

        J = greedy_optimizer(feature_dict, layer_idx, m_sharp, Z, tau)
        print(torch.__config__.parallel_info()) 

        # save the optimal J
        J_sharp.append(J)
        taus.append(tau)
        
        # update parameter for previous layer.
        #J_sharp.append(greedy_optimizer(feature_dict, idx, compressed_size[idx], Z, tau[idx]))
        W = model.linear_relu_stack[layer_idx*2].weight
    J_sharp.reverse()
    taus.reverse()

    # save J.
    d = {'J': J_sharp}
    with open('data/J.json', 'w') as f:  
        json.dump(d, f, indent=4)

    with open('data/J.json') as f:
        df = json.load(f)
    J_sharp = df['J']
    print(compressed_size)
    check_condition(compressed_size, tau_tildes, J_sharp)


    for layer_idx in range(layer_num):
        taus.append(compressed_size[layer_idx] * lambdas[layer_idx] * torch.tensor(tau_tildes[layer_idx]).to(device))
    compressed_model = construct(J_sharp, taus, compressed_size, original_model, feature_dict, device)
    return compressed_model, compressed_size

    lambdas = []
    compressed_size = []
    tau_tilde = []
    for layer_idx in range(layer_num):
        tau, lamb, m_sharp = est_hyper(feature_dict, layer_idx, original_size[layer_idx], device)
        tau_tilde.append(tau)
        compressed_size.append(m_sharp)
        lambdas.append(lamb)
    print(compressed_size)
    sys.exit()
    tau = [ compressed_size[idx] * lambdas[idx] * tau_tilde[idx] for idx in range(len(lambdas))]

    Z = None
    J = None
    J_sharp = []
    for idx in reversed(range(layer_num)):
        q = Q(tau_tilde[idx+1], idx, J) if idx < layer_num-1 else 0
        J_sharp.append(greedy_optimizer(feature_dict, idx, compressed_size[idx], Z, tau[idx]))
        W = model.linear_relu_stack[idx*2].weight
        J = J_sharp[-1]
        Z = z(original_size[idx], q, J, W)
    J_sharp.reverse()

    compressed_model = construct(J_sharp, tau, compressed_size)
    return compressed_model, compressed_size