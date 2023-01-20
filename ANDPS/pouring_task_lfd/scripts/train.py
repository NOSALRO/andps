from numpy import load
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import net_rd as net, CustomDataset
import numpy as np
eef_link_name = "panda_ee"

# read dataset from npy file (binary)
dataset = load("data/pour_demos.npy")
np_X = dataset[:, :6]
np_Y = dataset[:, -6:]

# Train
dim = np_X.shape[1]
target = np.zeros_like(np_X[-1,:])
target[0:6] = load('data/pour_demos_target.npy')[0:6]
num_DS = 4
net = net(dim, num_DS, target)
tensor_x = torch.Tensor(np_X)
tensor_y = torch.Tensor(np_Y)
batch_size = 128
dataset = CustomDataset(np_X, np_Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
batches_per_epoch = np_X.shape[0]//batch_size
epochs = 2000
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
loss_fn = nn.functional.mse_loss
lrs = []
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, output]
        inputs, outputs = data
        # print(inputs)
        batch_loss = 0.0
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output, _, _ = net(inputs.view(-1, dim), False)
        loss = torch.nn.functional.mse_loss(output, outputs.view(-1, dim), reduction='mean')
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_mean_loss = running_loss/batches_per_epoch
    print("Epoch: ", epoch+1, "Loss: ", train_mean_loss)
print('Finished Training')
torch.save(net.state_dict(),'pytorch models/lfd_all_euler.pt')
