import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import andps_images as andps,  CustomDataset
from torchvision import transforms
convert_tensor = transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read dataset
data_angle = np.load("data/angle.npz")
data_spiral = np.load("data/line.npz")
data_line = np.load("data/spiral.npz")

data = [data_angle]

# create dataset with image indexing
# images = []
targets = []
for i in range(1):

    targets.append(data[i]["eef_x"][-1,:])
    if(i == 0):
        np_X = torch.Tensor(data[i]["eef_x"])
        # flatten image
        images_flat = torch.Tensor(data[i]["images"].reshape(64*64,-1).T)
        np_X = torch.cat((np_X,images_flat),1)
        np_Y = torch.Tensor(data[i]["eef_vx"])
    else:
        cur_np_X = torch.Tensor(data[i]["eef_x"])
        images_flat = torch.Tensor(data[i]["images"].reshape(64*64,-1).T)
        cur_np_X = torch.cat((cur_np_X,images_flat),1)
        np_X = torch.cat((np_X, cur_np_X), 0)
        np_Y = torch.cat((np_Y, torch.Tensor(data[i]["eef_vx"])), 0)

print(np_X.shape)

assert np_X.shape[0]==1*720
assert np_X.shape[1]==(3+64*64)

target = np.zeros(3)
for t in targets:
    target+=t/1

dataset = CustomDataset(np_X, np_Y)

# Dataset now is in the form X = [[posX, posY, image_id],...]  Y = [[velX, velY]] (tensors)
# ds Matrix dimension
dim = np_Y.shape[1]
print(dim)
# print(np_X)



# Number of dynamical systems
num_DS = 6
net = andps(dim, num_DS, target, 3,device)
net.to(device)

# Hyperparameters
batch_size = 64
epochs = 200
le_r = 1e-3

# Load dataset

dataset = CustomDataset(np_X, np_Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

batches_per_epoch = np_X.shape[0]//batch_size
optimizer = torch.optim.AdamW(net.parameters(), lr=le_r)

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
        output = net(inputs.to(device).view(-1, 3+(64*64)))
        loss = torch.nn.functional.mse_loss(output, outputs.to(device).view(-1, dim), reduction='mean')
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_mean_loss = running_loss/batches_per_epoch
    print("Epoch: ", epoch+1, "Loss: ", train_mean_loss)
    if ((epoch + 1) % 10 == 0):
        torch.save(net.state_dict(),'models/panda_image_' + str(epoch + 1) + '.pt')
print('Finished Training')
torch.save(net.state_dict(), 'models/panda_image_' + str(epoch + 1) + '.pt')