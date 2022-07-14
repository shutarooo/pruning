import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
#import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import *
from loop import *
from dataloader import *

# train and save original network
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = ShallowNeuralNetwork().to(device)

    learning_rate = 1e-3
    batch_size = 600
    epochs = 10

    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader, test_dataloader = dataloader(batch_size)


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")

    PATH = 'data/original_shallow_model.pth'
    torch.save(model.state_dict(), PATH)

if __name__ == "__main__":
    main()