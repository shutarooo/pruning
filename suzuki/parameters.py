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

layer_num = 3
original_size = [300, 1000, 300, 10]
layer_keys = ['layer_1', 'layer_2', 'layer_3', 'out']
n = 60000

'''layer_num = 1
original_size = [1000, 10]
layer_keys = ['layer_1', 'out']
n = 60000'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cov_mx(feature_dict, layer, dim1, dim2):
    
    key = layer_keys[layer]
    if type(dim1)==int and type(dim2)==int:
        #print((torch.matmul(torch.t(feature_dict[key]), feature_dict[key])/n).device)
        return torch.matmul(torch.t(feature_dict[key]), feature_dict[key])/n

    elif type(dim1)==list and type(dim2)==list:
        J = torch.tensor(dim1).to(device)
        partial_tensor = torch.index_select(feature_dict[key], 1, J)
        #partial_tensor = feature_dict[key][:, dim1]
        return torch.t(partial_tensor) @ partial_tensor/n

    elif type(dim1)==list and type(dim2)==int:
        J = torch.tensor(dim1).to(device)
        partial_tensor = torch.index_select(feature_dict[key], 1, J)
        #partial_tensor = feature_dict[key][:, dim1]
        full_tensor = feature_dict[key]
        return torch.matmul(torch.t(partial_tensor), full_tensor)/n

    elif type(dim1)==int and type(dim2)==list:
        J = torch.tensor(dim2).to(device)
        partial_tensor = torch.index_select(feature_dict[key], 1, J)
        #partial_tensor = feature_dict[key][:, dim2]
        full_tensor = feature_dict[key]
        return torch.t(torch.matmul(torch.t(partial_tensor), full_tensor))/n

def convex_loss(feature_dict, layer, J, Z, tau_all, cov_ff):
    theta = 0.3
    m = original_size[layer]
    
    #print(cov_ff.size())
    #torch.inverse(cov_ff)
    #sys.exit()
    cov_fj = cov_mx(feature_dict, layer, m, J)
    cov_jf = torch.t(cov_fj)
    #cov_jf = cov_mx(feature_dict, layer, J, m)
    cov_jj = cov_mx(feature_dict, layer, J, J)

    tau = []
    for j in range(len(tau_all)):
        if j in J:
            tau.append(tau_all[j])
    tau = torch.tensor(tau).to(device)

    mx1 = cov_jj + torch.diag(tau).to(device)
    if torch.linalg.det(mx1) == 0:
        #print('singular.')
        return 10**8
    mx2 = cov_ff - cov_fj@torch.inverse(mx1)@cov_jf
    La = torch.trace(mx2)
    Lb = torch.trace( Z@mx2@torch.t(Z) )

    #La = input_loss(cov_ff, cov_fj, cov_jf, mx)
    #Lb = output_loss(cov_ff, cov_fj, cov_jf, Z, mx)

    return theta*La + (1-theta)*Lb

# no use
def input_loss(cov_ff, cov_fj, cov_jf, mx):
    
    return torch.trace(  )
# no use
def output_loss(cov_ff, cov_fj, cov_jf, Z, mx):
    return torch.trace( Z@(cov_ff - cov_fj@torch.inverse(mx)@cov_jf)@torch.t(Z) )

#以下2つの凸結合を返す
def N(theta, layer, Z, lamb, device, feature_dict):
    return theta * N_hat(layer, lamb, device, feature_dict) + (1-theta) * N_dash(layer, Z, device, feature_dict)

def N_hat(layer, lamb, device, feature_dict):
    #print(layer)
    m = original_size[layer]
    return torch.trace(cov_mx(feature_dict, layer, m, m) @ torch.inverse(cov_mx(feature_dict, layer, m, m) + lamb * torch.eye(original_size[layer]).to(device)))

def N_dash(layer, Z, device, feature_dict):
    m = original_size[layer]
    return torch.trace(Z @ cov_mx(feature_dict, layer, m, m) @ torch.inverse(cov_mx(feature_dict, layer, m, m) + torch.eye(original_size[layer]).to(device)) @ torch.t(Z))

# idxで指定されたlayerのqを返す
def Q(tau_tilde_plus, idx, J_plus):
    denom = 0
    for j in J_plus:
        denom += 1/tau_tilde_plus[j]
    print(len(tau_tilde_plus))
    #denom = [ torch.sum(tau_tilde[layer[i], J[i]]) for i in range(len(layers))]
    q = [ tau_tilde_plus[i]/denom if i in J_plus else 0 for i in range(original_size[idx+1])]
    #q = [ tau_tilde[layer[i+1], j] / denom[i+1] if j in J[i+1] else 0 for j in range(original_size[i]) for i in range(len(layers)-1)]
    return q

# 指定されたlayerのZを返す
def z(m, q, J_plus, W, device):
    #weight = params['linear_relu_stack.' + str(2*(l+1))+ '.weight']
    norm_max = 0
    for row in W:
        norm = torch.norm(row)
        if norm > norm_max:
            norm_max = norm
    
    diag = [math.sqrt(m * q[j]) / norm_max for j in J_plus]
    #diag = [math.square(original_size[l] * q[l][j]) / norm_max for j in J]
    diag_mx = torch.diag(torch.tensor(diag)).to(device)
    return diag_mx @ W[J_plus]