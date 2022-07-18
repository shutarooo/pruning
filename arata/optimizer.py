import torch
from torch import NoneType
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
#import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append('../')

from original.loop import *
from original.model import *

from suzuki.model import *

import sys

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, A_t, X, transform=None, target_transform=None):
        self.labels = A_t
        self.inputs = X
        print(A_t.size())
        print(X.size())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.size()[1]

    def __getitem__(self, idx):
        input = self.inputs[:,idx]
        label = self.labels[:,idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return input, label



def opt_tanh(A_t, X):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = TanhNetwork().to(device)

    learning_rate = 1e-3
    batch_size = 600
    epochs = 10

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_data = CustomDataset(A_t, X)
    train_dataloader = DataLoader(training_data, batch_size=600, shuffle=True)


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
    print("Done!")

    print(model.linear_relu_stack)

    W = model.linear_relu_stack[0].weight
    #A = W @ X
    #print(torch.norm(torch.inverse(W @ torch.t(W) @ W)))
    print(torch.norm(W))

    PATH = 'data/tanh_weight.pth'
    torch.save(model.state_dict(), PATH)

    return model.linear_relu_stack[0].weight
