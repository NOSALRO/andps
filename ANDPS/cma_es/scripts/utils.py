def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


import numpy as np
import RobotDART as rd
import torch
import torch.nn as nn
import geotorch
from torch.utils.data import Dataset
import torch.nn.functional as F

# ----------- PyTorch -----------

# Custom Dataset class used for accessing dataset through pytorch dataloaders


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


class net_rd(nn.Module):
    def __init__(self, ds_dim, N, target, device='cpu'):
        super(net_rd, self).__init__()
        self.N = N
        self.ds_dim = ds_dim
        self.n_params = ds_dim

        self.all_params_B_A = nn.ModuleList([nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.positive_semidefinite(self.all_params_B_A[i])

        self.all_params_C_A = nn.ModuleList([nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.skew(self.all_params_C_A[i])

        self.all_weights = nn.Sequential(
            nn.Linear(self.ds_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, N), nn.Softmax(dim=1))
        self.x_tar = torch.Tensor(target).view(-1, ds_dim).to(device)

    def forward(self, x, disp=False):
        batch_size = x.shape[0]
        s_all = torch.zeros((1, self.ds_dim))
        w_all = torch.zeros((batch_size, self.N))

        w_all = self.all_weights(x)
        #w_all = torch.nn.functional.softmax(w_all, dim=1)
        if disp:
            print(w_all)

        for i in range(self.N):
            A = (self.all_params_B_A[i].weight + self.all_params_C_A[i].weight)

            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(A, (self.x_tar-x).transpose(0, 1)).transpose(0, 1))
        return s_all

class simple_nn(nn.Module):
    def __init__(self, dim):
        super(simple_nn, self).__init__()
        self.fc1 = nn.Linear(dim, 28)
        self.fc2 = nn.Linear(28, 28)
        self.fc3 = nn.Linear(28, 28)
        self.fc4 = nn.Linear(28, dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ----------- Robot Dart -----------
class PIDTask:
    def __init__(self, dt, Kp=10., Ki=0.01, Kd=0.1):
        self._target = None
        self._dt = dt
        self._Kp = Kp
        self._Kd = Kd
        self._Ki = Ki
        self._sum_error = 0
        self._prev_error = None

    def set_target(self, target):
        self._target = target
        self._sum_error = 0
        self._prev_error = None

    def error(self, tf):
        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]

    def update(self, current):
        error_in_world_frame = self.error(current)
        if self._prev_error is None:
            derror = np.zeros_like(error_in_world_frame)
        else:
            derror = (error_in_world_frame - self._prev_error) / self._dt
        self._prev_error = error_in_world_frame

        self._sum_error = self._sum_error + error_in_world_frame * self._dt
        return self._Kp * error_in_world_frame + self._Ki * self._sum_error + self._Kd * derror, self.target_reached(error_in_world_frame)

    def target_reached(self, error):
        if np.linalg.norm(error[3:]) < 0.02:
            return True
        return False

    def set_gains(self, Kp, Ki, Kd):
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd


class PIJoint:
    def __init__(self, dt, target=None, Kp=10., Ki=0.1):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0.

    def set_target(self, target):
        self._target = target

    def update(self, current):
        # since we have angles, it's better to wrap into [-pi,pi)
        error = angle_wrap_multi(self._target - current)
        self._sum_error = self._sum_error + error * self._dt
        return self._Kp * error + self._Ki * self._sum_error


def angle_wrap(theta):
    while theta < -np.pi:
        theta += 2 * np.pi
    while theta > np.pi:
        theta -= 2 * np.pi
    return theta


def angle_wrap_multi(theta):
    if isinstance(theta, list):
        th = theta
        for i in range(len(th)):
            th[i] = angle_wrap(th[i])
        return th
    elif type(theta) is np.ndarray:
        th = theta
        for i in range(theta.shape[0]):
            th[i] = angle_wrap(th[i])
        return th
    return angle_wrap(theta)


def damped_pseudoinverse(jac, l=0.1):
    m, n = jac.shape
    if n >= m:
        return jac.T @ np.linalg.inv(jac @ jac.T + l*l*np.eye(m))
    return np.linalg.inv(jac.T @ jac + l*l*np.eye(n)) @ jac.T



def EREulerXYZ(eulerXYZ):
    x = eulerXYZ[0]
    y = eulerXYZ[1]
    # z = eulerXYZ[2]
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)

    R = np.zeros((3, 3))
    R[0, 0] = 1.
    R[0, 2] = sin_y
    R[1, 1] = cos_x
    R[1, 2] = -cos_y * sin_x
    R[2, 1] = sin_x
    R[2, 2] = cos_x * cos_y

    return R

