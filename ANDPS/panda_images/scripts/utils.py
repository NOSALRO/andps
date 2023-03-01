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




class CNN_lasa_image(nn.Module):
    def __init__(self, classes, dim, N):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(1690+dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, N)

    def forward(self, state):
        # batch of images
        # img = torch.empty(size=(state.shape[0], 1, 64, 64)).to(state.device)
        # for i in range(state.shape[0]):
        #         img[i] = state[i,3:].clone().detach().reshape(64,64)
        #         # import matplotlib.pyplot as plt
        #         # plt.imshow(img[i][0].cpu().numpy(),cmap='gray')
        #         # plt.show()

        x = self.pool1(F.relu(self.conv1(state[:,3:].view(state.shape[0],1,64,64))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.shape)
        # print(x)
        x = torch.cat((x, state[:, :3]), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)
        return x


class andps_images(nn.Module):
    def __init__(self, ds_dim, N, target, classes,  device='cpu'):
        super().__init__()
        self.N = N
        self.ds_dim = ds_dim
        self.n_params = ds_dim
        self.all_weights = CNN_lasa_image(classes, self.ds_dim, N).to(device)
        self.all_params_B_A = nn.ModuleList([nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])
        # self.images = images
        for i in range(N):
            geotorch.positive_semidefinite(self.all_params_B_A[i])

        self.all_params_C_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.skew(self.all_params_C_A[i])
        self.x_tar = torch.Tensor(target).view(-1, ds_dim).to(device)



    def forward(self, x):
        batch_size = x.shape[0]
        s_all = torch.zeros((1, self.ds_dim)).to(x.device)
        w_all = self.all_weights(x)
        print(np.mean(w_all.detach().cpu().numpy(),axis=0))
        for i in range(self.N):
            A = self.all_params_B_A[i].weight + self.all_params_C_A[i].weight

            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(A, (self.x_tar-x[:, :3]).transpose(0, 1)).transpose(0, 1))
        return s_all


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
