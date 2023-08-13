#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from nn_utils import go1_images as net
from nn_utils import CustomDataset as CustomDataset
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
convert_tensor = transforms.ToTensor()



# read data
dataset = np.load("data/test.npz")

np_X = dataset["X"]
np_Y = dataset["Y"]
target = dataset["mean_target"]

dim = np_Y.shape[1]
whole_state = np_X.shape[1]
num_DSs = 3

net = net(dim, num_DSs, target, device).to(device)

batch_size = 256
epochs = 1000
le_r = 1e-3


dataset = CustomDataset(np_X, np_Y)
dataset = CustomDataset(np_X, np_Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

batches_per_epoch = np_X.shape[0]//batch_size
optimizer = torch.optim.AdamW(net.parameters(), lr=le_r, weight_decay=0.1)
# loss_fn = nn.functional.mse_loss
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias)
net.apply(weights_init)

for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, datas in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, output]
        inputs, outputs = datas
        # print(inputs)
        batch_loss = 0.0
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = net(inputs.to(device).view(-1, whole_state))
        # print(output.shape, outputs.view(dim, -1).shape)
        # + 0.005 * (1-w_all).square().sum().sum()
        loss = torch.nn.functional.mse_loss(
            output, outputs.to(device).view(-1, dim), reduction='mean')
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss.item())
    # scheduler.step()
    train_mean_loss = running_loss/batches_per_epoch
    # print(i)
    print("Epoch: ", epoch+1, "Loss: ", train_mean_loss)
    if ((epoch + 1) % 10 == 0):
        torch.save(net.state_dict(),'models/test' + str(epoch + 1) + '.pt')
print('Finished Training')
torch.save(net.state_dict(),'models/test' + str(epoch + 1) + '.pt')