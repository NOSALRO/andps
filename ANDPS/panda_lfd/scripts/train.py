import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import net_rd as andps,  CustomDataset, simple_nn


experiment = "eef"
# experiment = "joint"
# policy = "ANDPS"
policy = "NN"

# read dataset
dataset = np.load("data/spiral_" + experiment + "_space.npy")

if experiment == "eef":
    np_X = dataset[:, :3]
    np_Y = dataset[:, -3:]

elif experiment == "joint":
    np_X = dataset[:, :7]
    np_Y = dataset[:, -7:]

dataset = CustomDataset(np_X, np_Y)

# setup policy
dims = np_X.shape[1]
target = np_X[-1,:]
num_DS = 6
if(policy == "ANDPS"):
    model = andps(dims, num_DS, target)
elif(policy == "NN"):
    model = simple_nn(dims)

# hyperparameters
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
batches_per_epoch = np_X.shape[0]//batch_size
epochs = 1000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, output]
        inputs, outputs = data
        batch_loss = 0.0
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(inputs.view(-1, dims))
        loss = torch.nn.functional.mse_loss(output, outputs.view(-1, dims), reduction='mean')
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_mean_loss = running_loss/batches_per_epoch
    print("Epoch: ", epoch+1, "Loss: ", train_mean_loss)
print('Finished Training')

torch.save(model.state_dict(),'models/'+policy+'_spiral_'+experiment+'_1000.pt')
