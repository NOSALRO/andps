import numpy as np
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

# Lasa simple weight module / first order / known target
class net_first_order(nn.Module):
    def __init__(self, ds_dim, N, target, device='cpu'):
        super(net_first_order, self).__init__()
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
            nn.Linear(self.ds_dim, 10), nn.ReLU(), nn.Linear(10, N), nn.Softmax(dim=1))
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

class simple_nn(nn.Module):
    def __init__(self, dim):
        super(simple_nn, self).__init__()
        self.fc1 = nn.Linear(dim, 28) # prev 20

        self.fc4 = nn.Linear(28, dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Lasa one hot weight module / first order / known target
class net_one_hot(nn.Module):
    def __init__(self, ds_dim, N, target, num_of_one_hots, device='cpu'):
        super().__init__()
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
            nn.Linear(self.ds_dim+num_of_one_hots, 32), nn.ReLU(),nn.Linear(32, N), nn.Softmax(dim=1))

        self.x_tar = torch.Tensor(target).view(-1, 2).to(device)

    def forward(self, x, disp=True):
        batch_size = x.shape[0]
        s_all = torch.zeros((1, self.ds_dim)).to(x.device)
        # w_all = torch.zeros((batch_size, self.N))
        w_all = self.all_weights(x)
        # w_all = torch.nn.functional.softmax(w_all, dim=1)
        if disp:
            print(w_all)
        reA = []
        for i in range(self.N):
            A = (self.all_params_B_A[i].weight + self.all_params_C_A[i].weight)
            reA.append(A)
            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(
                A, (self.x_tar-x[:, :2]).transpose(0, 1)).transpose(0, 1))
        return s_all, w_all, reA

# Lasa cnn weight module / first order / known target
class CNN_lasa_image(nn.Module):
    def __init__(self, classes, dim, N, images):
        super().__init__()
        self.images = [img.detach() for img in images]
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5,5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5,5))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))
        self.fc1 = nn.Linear(1960+dim, 500)
        #self.bn = nn.BatchNorm1d(1960+dim)
        self.fc2 = nn.Linear(500, N)

    def forward(self, state):
        # batch of images
        img = torch.empty(size=(state.shape[0], 1, 68, 68)).to(state.device)
        for i in range(state.shape[0]):
            img[i] = self.images[int(state[i, 2])].clone().detach()

        x = self.pool1(F.relu(self.conv1(img)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.shape)
        # print(x)
        x = torch.cat((x, state[:, :2]), dim=1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)


        return x

class lasa_images_net_fo(nn.Module):
    def __init__(self, ds_dim, N, target, classes, images, device='cpu'):
        super().__init__()
        self.N = N
        self.ds_dim = ds_dim
        self.n_params = ds_dim
        self.all_weights = CNN_lasa_image(classes, self.ds_dim, N, images).to(device)
        self.all_params_B_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])
        # self.images = images
        for i in range(N):
            geotorch.positive_semidefinite(self.all_params_B_A[i])

        self.all_params_C_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.skew(self.all_params_C_A[i])
        self.x_tar = torch.Tensor(target).view(-1, ds_dim).to(device)

    def forward(self, x, disp=True):
        batch_size = x.shape[0]
        s_all = torch.zeros((1, self.ds_dim)).to(x.device)
        w_all = self.all_weights(x)
        if disp:
            print(w_all)
        reA = []
        for i in range(self.N):
            A = (self.all_params_B_A[i].weight + self.all_params_C_A[i].weight)
            reA.append(A)
            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(
                A, (self.x_tar-x[:, :2]).transpose(0, 1)).transpose(0, 1))
        return s_all, w_all, reA

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------ Dataset -------------
def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))

# Error
def compute_error(X, Y_train, Y_model, r = 0.6, q = 0.4, eps = 1e-20):
    assert(X.shape[0] == Y_train.shape[0] and X.shape[1] == Y_train.shape[1])
    assert(X.shape[0] == Y_model.shape[0] and X.shape[1] == Y_model.shape[1])

    e = 0.

    N = X.shape[0]

    for n in range(N):
        y_hat = Y_model[n, :]
        n_y_hat = np.linalg.norm(y_hat)

        y_gt = Y_train[n, :]
        n_y_gt = np.linalg.norm(y_gt)

        y_diff = y_gt - y_hat

        alpha = r * (1. - y_gt.dot(y_hat) / (n_y_gt * n_y_hat + eps)) ** 2

        beta = q * (y_diff.dot(y_diff)) / (n_y_gt * n_y_gt + eps)

        e_loc = np.sqrt(alpha + beta)

        e += e_loc

    e = e / float(N)

    return e

def n_choose_k_dataset_split(total_demos, test_num):
    assert(test_num <= total_demos and test_num > 0)
    arr = [i for i in range(total_demos)]

    combi = combinations(arr, test_num)

    train_idxs = []
    test_idxs = []

    arr_set = set(arr)
    for c in combi:
        c_set = set(c)
        rest = arr_set - c_set

        train_idxs.append(list(rest))
        test_idxs.append(list(c_set))

    return train_idxs, test_idxs

# Train test and val are unique
def train_test_val_split(seed, data, train_num=4, val_num=2, test_num=1):
    if(train_num+val_num+test_num != 7):
        print("Split values must add up to 7")
        return -1

    np.random.seed(seed)
    idxs = [0, 1, 2, 3, 4, 5, 6]

    # train
    train_ids = []
    if train_num == 0:
        X_train, Y_train = None, None
    else:
        for i in range(train_num):

            id = np.random.randint(0,len(idxs))
            train_ids.append(idxs[id])
            idxs.pop(id)
            X_train, Y_train = lasa_to_numpy(data, train_ids)

    # validation
    val_ids = []
    if val_num == 0:
        X_val, Y_val = None, None
    else:
        for i in range(0,val_num):
            id = np.random.randint(0,len(idxs))
            val_ids.append(idxs[id])
            idxs.pop(id)
            X_val, Y_val = lasa_to_numpy(data, val_ids)


    # test
    test_ids = []
    if train_num == 0:
        X_test, Y_test = None, None
    else:
        for i in range(len(idxs)):
            id = np.random.randint(0,len(idxs))
            test_ids.append(idxs[id])
            idxs.pop(id)
            X_test, Y_test = lasa_to_numpy(data, test_ids)

    print(train_ids)
    print(val_ids)
    print(test_ids)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# lasa dataset to numpy array (smoothed position and vels are calulated as ddot)
def lasa_to_numpy(data, ids=[0,1,2,3,4,5,6], smooth_val = 5):
    demos = data.demos
    demo_0 = demos[ids[0]]
    pos = np.array([smooth(demo_0.pos[0], smooth_val),  smooth(demo_0.pos[1], smooth_val)])
    dt_global = demo_0.dt
    vel = (pos[:, 1:]-pos[:, :-1])/dt_global
    pos = pos[:, 1:]
    for i in range(1, len(ids)):
            demo_i = demos[ids[i]]
            i_pos = np.array([smooth(demo_i.pos[0], smooth_val), smooth(demo_i.pos[1], smooth_val)])
            i_vel = (i_pos[:, 1:]-i_pos[:, :-1])/dt_global
            i_acc = demo_i.acc  # / 1000.
            i_pos = i_pos[:, 1:]
            pos = np.concatenate((pos, i_pos), axis=1)

            vel = np.concatenate((vel, i_vel), axis=1)
    np_X = pos.transpose()
    np_Y = vel.transpose()
    return np_X, np_Y