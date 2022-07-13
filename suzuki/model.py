import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
#import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CompressedNeuralNetwork(nn.Module):
    def __init__(self, weight, bias, compressed_size):
        super(CompressedNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, compressed_size[0]),
            nn.ReLU(),
            nn.Linear(compressed_size[0], compressed_size[1]),
            nn.ReLU(),
            nn.Linear(compressed_size[1], compressed_size[2]),
            nn.ReLU(),
            nn.Linear(compressed_size[2], 10),
            nn.ReLU()
        )
        for i in range(4):
            self.linear_relu_stack[i*2].weight = nn.Parameter(weight[i])
            self.linear_relu_stack[i*2].bias = nn.Parameter(bias[i])

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class LoadNeuralNetwork(nn.Module):
    def __init__(self, compressed_size):
        super(LoadNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, compressed_size[0]),
            nn.ReLU(),
            nn.Linear(compressed_size[0], compressed_size[1]),
            nn.ReLU(),
            nn.Linear(compressed_size[1], compressed_size[2]),
            nn.ReLU(),
            nn.Linear(compressed_size[2], 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
