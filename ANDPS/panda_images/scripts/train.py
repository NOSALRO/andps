import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import andps_images as andps,  CustomDataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
convert_tensor = transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read dataset
data_angle = np.load("data/angle.npz")
data_line = np.load("data/line.npz")
data_sine = np.load("data/sine.npz")

data = [data_angle, data_line, data_sine]

# create dataset with image indexing
# images = []
targets = []
for i in range(len(data)):
    targets.append(data[i]["eef_x"][-1,:])
    if(i == 0):
        np_X = torch.Tensor(data[i]["np_X"])
        np_Y = torch.Tensor(data[i]["eef_vx"])
    else:
        cur_np_X = torch.Tensor(data[i]["np_X"])
        np_X = torch.cat((np_X, cur_np_X), 0)
        np_Y = torch.cat((np_Y, torch.Tensor(data[i]["eef_vx"])), 0)


assert np_X.shape[0]==len(data)*720
assert np_X.shape[1]==(3+64*64)

target = np.zeros(3)
for t in targets:
    target+=t/len(data)

dataset = CustomDataset(np_X, np_Y)



dim = np_Y.shape[1]

# Number of dynamical systems
num_DS = 8
net = andps(dim, num_DS, target, device)
net.to(device)

# Hyperparameters
batch_size = 128
epochs = 200
le_r = 5e-4


# Load dataset

dataset = CustomDataset(np_X, np_Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

batches_per_epoch = np_X.shape[0]//batch_size
optimizer = torch.optim.AdamW(net.parameters(), lr=le_r, weight_decay=0.01)
# scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
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
    # scheduler.step()
    train_mean_loss = running_loss/batches_per_epoch
    print("Epoch: ", epoch+1, "Loss: ", train_mean_loss)#, "Lr: ", scheduler.get_lr()[0])
    if ((epoch + 1) % 10 == 0):
        torch.save(net.state_dict(),'models/panda_image_' + str(epoch + 1) + '.pt')
print('Finished Training')
torch.save(net.state_dict(), 'models/panda_image_' + str(epoch + 1) + '.pt')