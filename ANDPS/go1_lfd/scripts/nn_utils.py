from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import geotorch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

convert_tensor = transforms.ToTensor()

class CNN_go1_image(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 10 * 10 + 5, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, N)

    def forward(self, state):

        # only the image goes through the CNN
        imgs = state[:, 5:]
        # print(imgs.shape)

        # reshape the images so that they can be fed into the CNN (85x85) rgb, take batch size into account
        imgs = imgs.reshape(-1, 85, 85, 3)
        # print(imgs[0])
        # plt.imshow(imgs[0, :, :, :].cpu().numpy().astype(int))
        # plt.show()
        imgs = imgs.permute(0, 3, 1, 2)

        x = self.pool1(self.relu1(self.conv1(imgs)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = torch.flatten(x, 1)

        # the rest of the state has to pass through the fc layers aswell
        x = torch.cat([x, state[:, :5]], dim=1)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        x = F.softmax(x, dim =1)

        return x
# Simple andps module x = x_c + I


class go1_images(nn.Module):
    def __init__(self, ds_dim, N, target,  device='cpu', flag = True):
        super().__init__()

        # Number of dynamical systems
        self.N = N

        # Dimension of the dynamical system (Dimension of controllable part of the state)
        self.ds_dim = ds_dim


        # Whole state dependant weighing function
        self.all_weights = CNN_go1_image(N).to(device)


        self.all_params_B_A = nn.ModuleList([nn.Linear(self.ds_dim, self.ds_dim, bias=False) for i in range(N)])
        for i in range(N):
            geotorch.positive_semidefinite(self.all_params_B_A[i])

        self.all_params_C_A = nn.ModuleList([nn.Linear(self.ds_dim, self.ds_dim, bias=False) for i in range(N)])
        for i in range(N):
            geotorch.skew(self.all_params_C_A[i])

        self.all_params_P_A = nn.ModuleList([nn.Linear(self.ds_dim, self.ds_dim, bias=False)])
        geotorch.positive_definite(self.all_params_P_A[0])

        self.x_tar = torch.Tensor(target).view(-1, ds_dim).to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        s_all = torch.zeros((1, self.ds_dim)).to(x.device)
        w_all = self.all_weights(x)
        # print(np.average(w_all.cpu().detach().numpy(),axis=1))
        reA = []
        for i in range(self.N):
            A = torch.mul(self.all_params_P_A[0].weight, (self.all_params_B_A[i].weight + self.all_params_C_A[i].weight))
            reA.append(A)
            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(
                A, (self.x_tar-x[:, :3]).transpose(0, 1)).transpose(0, 1))
        return s_all

class CustomDataset(Dataset):
    """ Custom Linear Movement Dataset. """

    def __init__(self, x, y):
        """
        Args:
            x (numpy array): # of demos x controllable vec size numpy array that contains x_c (INPUTS)
            y (numpy array): # of demos x controllable dvec/dt  size numpy array that contains x_cdot (OUTPUTS)
        """
        self.x = torch.Tensor(x)

        # self.x = torch.Tensor(x[:,:5])
        # imgs = x[:,5:].reshape(-1, 3, 85, 85)
        # imgs_tensor = torch.zeros((imgs.shape[0], 85, 3, 85))
        # for i in range(imgs.shape[0]):
        #     imgs_tensor[i] = convert_tensor(imgs[i])

        # # flatten image and cat to x
        # self.x = torch.cat([self.x, imgs_tensor.reshape(-1, 3*85*85)], dim=1)


        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

