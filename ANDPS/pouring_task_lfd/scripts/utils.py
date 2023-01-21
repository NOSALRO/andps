import numpy as np
import RobotDART as rd
import torch
import torch.nn as nn
import geotorch
from torch.utils.data import Dataset
import torch.nn.functional as F
from itertools import combinations

# ----------- Pytorch -----------
# Custom Dataset class used for accessing lasa dataset through pytorch dataloaders
class CustomDataset(Dataset):
    """ Custom Linear Movement Dataset. """

    def __init__(self, x, y):
        """
        Args:
            x (numpy array): # of demos x controllable vec size numpy array that contains x_c (INPUTS)
            y (numpy array): # of demos x controllable dvec/dt  size numpy array that contains x_cdot (OUTPUTS)
        """
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Robot_dart simple weight module / first order / known target
class net_rd(nn.Module):
    def __init__(self, ds_dim, N, target, device='cpu'):
        super(net_rd, self).__init__()
        self.N = N
        self.ds_dim = ds_dim
        self.n_params = ds_dim

        self.all_params_B_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.positive_semidefinite(self.all_params_B_A[i])

        self.all_params_C_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.skew(self.all_params_C_A[i])

        self.all_weights = nn.Sequential(
            nn.Linear(self.ds_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, N), nn.Softmax(dim=1))
        self.x_tar = torch.Tensor(target).view(-1, ds_dim).to(device)

    def forward(self, x, disp=True):
        batch_size = x.shape[0]
        s_all = torch.zeros((1, self.ds_dim)).to(x.device)
        # w_all = torch.zeros((batch_size, self.N))

        w_all = self.all_weights(x)
        #w_all = torch.nn.functional.softmax(w_all, dim=1)
        if disp:
            print(w_all)
        reA = []
        for i in range(self.N):
            A = (self.all_params_B_A[i].weight + self.all_params_C_A[i].weight)
            reA.append(A)
            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(
                A, (self.x_tar-x).transpose(0, 1)).transpose(0, 1))
        return s_all, w_all, reA

