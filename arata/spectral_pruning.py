from curses import A_DIM, A_TOP
from re import X
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
from optimizer import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''layer_num = 3
original_size = [300, 1000, 300, 10]
layer_keys = ['layer_1', 'layer_2', 'layer_3', 'out']
n = 60000'''

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

'''def opt_tanh(a, X_t):
    w = torch.zeros(X_t.size()[1])
    grad = torch.zeros(X_t.size()[1])
    for i in range(a.size()[0]):
        grad += a[i]-'''


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
        '''At = A_list[layer_idx-1]             
        A = torch.t(A_list[layer_idx-1])  '''   
        A = A_list[layer_idx-1]
        At = torch.t(A)
        P = torch.t(feature_dict[layer_keys[layer_idx-1]]) @ A @ torch.inverse(At@A)
        print('P: {}'.format(P.size()))

        # Estimate bias.
        X_t = None
        if layer_idx == 1:
            tmp = extract_loader.__iter__()
            X_t, target = tmp.next()
            X_t = torch.flatten(X_t, 1, -1).to(device)
        else:
            X_t = feature_dict[layer_keys[layer_idx-2]]
        apx = P @ ReLU(W_list[layer_idx-1] @ torch.t(X_t))      #

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
    
def pseudo_inv(layer_idx, extract_loader, feature_dict):
    X_t = None
    if layer_idx == 0:
        tmp = extract_loader.__iter__()
        X_t, target = tmp.next()
        X_t = torch.flatten(X_t, 1, -1).to(device)
    else:
        X_t = feature_dict[layer_keys[layer_idx-1]]
    X_t_rank = torch.linalg.matrix_rank(X_t)
    print(torch.linalg.matrix_rank(X_t))
    # select linearly independent rows from X_t.
    
    X_idp = torch.unsqueeze(X_t[0], 0)
    current_rank = 1
    for i in range(1, X_t.size()[0]):
        X_column = torch.unsqueeze(X_t[i], 0)
        X_idp = torch.cat((X_idp, X_column), 0)
        if current_rank == torch.linalg.matrix_rank(X_idp):
            X_idp = X_idp[:-1]
        elif current_rank == X_t_rank:
            break
    pseudo_X = torch.t(X_idp) @ torch.inverse(X_idp @ torch.t(X_idp))

    return pseudo_X, X_t

def gaussian_elimination(X):
    #print(torch.sum(X, dim=1))
    #Xa = torch.cat((X,a), 1)
    Xa = X
    #print(Xa)
    j = 0
    finish_flag = False

    history_mx = torch.eye(Xa.size()[0])

    for i in range(Xa.size()[0]):
        #print(i,j)
        while Xa[i][j]==0:
            # check columns. if there is non-zero element, swap rows.
            all_zero_flag = True
            for k in range(i, Xa.size()[0]):
                if Xa[k][j] != 0:
                    #print('Xa[i][k]: {}'.format(Xa[i][k]))
                    all_zero_flag = False
                    Xa[[i,k]] = Xa[[k,i]]
                    break
            if all_zero_flag == False:
                #print('swap')
                break
            # if all element is zero, change column.
            if all_zero_flag == True:
                #print('change column')
                j+=1
                # if this is a final column, elimination is done.
                if j==Xa.size()[1]-1:
                    finish_flag = True
                    break
                else:
                    continue
            
        if finish_flag:
            break

        # normalize target row.
        norm_mx = torch.eye(Xa.size()[0]).to(device)
        norm_mx[i][i] = 1/Xa[i][j]
        Xa = norm_mx @ Xa
        history_mx = norm_mx @ history_mx
        #print(Xa)

        # eliminate columns.
        column = Xa[:,j]
        elim_mx = torch.eye(Xa.size()[0]).to(device)
        elim_mx[:,i] = -column
        Xa = elim_mx @ Xa
        history_mx = elim_mx @ history_mx

        #print(Xa.int())

        j += 1
        if j==Xa.size()[1]-1:
            break

    return Xa, history_mx






def compress(original_model, compressed_size, feature_dict, extract_loader):
    
    ReLU = torch.nn.ReLU()
    W_list = []
    A_list = []
    for layer_idx in range(layer_num):
        print('@@@@@@@@@@ EVD, calculation. @@@@@@@@@@@@@')
        print("------- {} layer --------".format(layer_idx))

        #X_tri, history_mx = gaussian_elimination(X)
        #pseudo_X, X_t = pseudo_inv(layer_idx, extract_loader, feature_dict)
        tmp = extract_loader.__iter__()
        X_t, target = tmp.next()
        X_t = torch.flatten(X_t, 1, -1).to(device)

        # EV information
        e_values, e_vectors = EVD(feature_dict, layer_idx)
        print('e_vector.len: {}'.format(e_vectors.size()[0]))

        for i in range(1000):
            cnt=0
            #print(torch.max(e_vectors[:,i]))
            #print(torch.min(e_vectors[:,i]))
            #print(torch.inner(e_vectors[:,0], e_vectors[:,i]))
            e_vectors[:,i] = e_vectors[:,i]/torch.linalg.norm(e_vectors[:,i])
            
            for j in range(e_vectors.size()[0]):
                c=1
                
        #sys.exit()

        A = None
        W = None
        print(torch.linalg.matrix_rank(X_t))
        for j in range(int((compressed_size[layer_idx]+1)/2)):
            #a = e_vectors[:,j] / torch.linalg.norm(e_vectors[:,j]) *10**2
            A = e_vectors

            '''W = opt_tanh(A_t[:500],torch.t(X_t))
            A = torch.t(A_t[:500])'''
            #W = opt_tanh(A_t,torch.t(X_t))

            loss = 0
            Sigma = torch.t(feature_dict['layer_1'])
            Proj_A = A @ torch.linalg.inv(torch.t(A)@A) @ torch.t(A)
            for sigma in Sigma:
                loss += torch.norm(sigma - sigma @ Proj_A )
            print(loss)
            sys.exit()
            break
            #sys.exit()

            
            #w_plus = pseudo_X @ a
            W_plus_info = torch.linalg.lstsq(X_t, torch.t(a).to(device))
            
            w_plus = W_plus_info.solution
            print('residuals: {}'.format(torch.dist(w_plus, torch.linalg.pinv(X_t) @ torch.t(a))))
            #W_minus = -1 * W_plus
            w_minus = -w_plus
            continue

            a_plus = ReLU(a)
            a_minus = ReLU(-1 * a)

            print('LE solution is exact?  {}'.format(a==X_t @ w_plus))
            if A == None:
                if torch.max(a_plus) <= 0:
                    A = torch.unsqueeze(a_minus, 0)
                else:
                    A = torch.unsqueeze(a_plus, 0)
            else:
                A = torch.cat((A, torch.unsqueeze(a_plus, 0)), 0)
                A = torch.cat((A, torch.unsqueeze(a_minus, 0)), 0)

            if W == None:
                if torch.max(a_plus) <= 0:
                    W = torch.unsqueeze(w_minus, 0)
                else:
                    W = torch.unsqueeze(w_plus, 0)
            else:
                W = torch.cat((W, torch.unsqueeze(w_plus, 0)), 0)
                W = torch.cat((W, torch.unsqueeze(w_minus, 0)), 0)
            print(A.size())
            print(W.size())

            #print(j, w_plus.size(), w_plus_info.residuals)
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

    #print(X_t.device)
    #print(e_vector.device) 

    # SVD information
    '''U, S, V_t = torch.linalg.svd(X_t, full_matrices=False)
    S_inv = torch.tensor([1/s if s!=0 else s for s in S]).to(device)'''

    '''print(X_idp.size())
                X_column = torch.unsqueeze(X_t[i], 0)
                print(torch.unsqueeze(X_t[i], 0).size())
                cat_tens = torch.cat((X_idp, X_column), 0)
                print(cat_tens.size())
                sys.exit()
                
        #print(torch.t(a).to(device).size())
            x = torch.unsqueeze(torch.t(a).to(device), 0)
            #print(x.size())
            cat_tens = torch.cat((torch.t(x), X_t), 1)
            #print(cat_tens.size())
            print(torch.linalg.matrix_rank(X_t))
            print(torch.linalg.matrix_rank(cat_tens))
            continue'''




    #sys.exit()
    '''W_plus_info = torch.linalg.lstsq(X_t, torch.t(e_vector).to(device))
    W_plus = W_plus_info.solution
    W_minus = -1 * W_plus'''

    '''x = torch.unsqueeze(torch.t(a).to(device), 0)
            print(torch.t(x).size())
            cat_tens = torch.cat((X_t, torch.t(x)), 1)
            print(cat_tens.size())
            
            print(torch.linalg.matrix_rank(cat_tens))

            continue'''


