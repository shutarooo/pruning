from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
#import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#from dataloader import *


def train_loop(dataloader, model, loss_fn, optimizer, device, is_save=False):
    size = len(dataloader.dataset)
    correct = 0

    for batch, (X, y) in enumerate(dataloader):       
        X = X.to(device)
        y = y.to(device) 
        #print(X.device)
        # 予測と損失の計算
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if batch % 10 == 0:
            #correct /= size
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")
    
    #if is_save:
        

        


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device) 
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    #if is_save:



def feature_extractor(model, extract_loader, device):
    return_layers = {
        'linear_relu_stack.1' : 'layer_1',
        'linear_relu_stack.3' : 'layer_2',
        'linear_relu_stack.5' : 'layer_3', 
        'linear_relu_stack.7' : 'out',
    }

    feature_extractor = create_feature_extractor(model, return_layers)

    with torch.no_grad():
        for X, y in extract_loader:
            X = X.to(device)
            #y = y.to(device) 
            feature_dict = feature_extractor(X)
    return feature_dict
    #feature_dict = feature_extractor(training_data[0][0].to(device))

def shallow_feature_extractor(model, extract_loader, device):
    return_layers = {
        'linear_relu_stack.1' : 'layer_1',
        'linear_relu_stack.3' : 'out',
    }

    feature_extractor = create_feature_extractor(model, return_layers)

    with torch.no_grad():
        for X, y in extract_loader:
            X = X.to(device)
            #y = y.to(device) 
            feature_dict = feature_extractor(X)
    return feature_dict
    #feature_dict = feature_extractor(training_data[0][0].to(device))
