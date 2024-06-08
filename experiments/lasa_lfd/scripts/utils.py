import numpy as np
import torch.nn as nn

import torch.nn.functional as F
from itertools import combinations

# ----------- Pytorch -----------


class simple_nn(nn.Module):
    def __init__(self, dim):
        super(simple_nn, self).__init__()
        self.fc1 = nn.Linear(dim, 28)
        self.fc4 = nn.Linear(28, dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------ Dataset -------------


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'validn')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def compute_error(x, y_train, y_model, r=0.6, q=0.4, eps=1e-20):
    """
    Computes the error between the ground truth and model-predicted trajectories.

    Parameters:
    - x (np.ndarray): Input data array.
    - y_train (np.ndarray): Ground truth trajectories.
    - y_model (np.ndarray): Model-predicted trajectories.
    - r (float): Weighting factor for the direction error term.
    - q (float): Weighting factor for the magnitude error term.
    - eps (float): Small value to prevent division by zero.

    Returns:
    - float: The computed error.
    """
    assert x.shape == y_train.shape == y_model.shape, "Input arrays must have the same shape."

    total_error = 0.0
    N = x.shape[0]

    for n in range(N):
        y_hat = y_model[n, :]
        n_y_hat = np.linalg.norm(y_hat)

        y_gt = y_train[n, :]
        n_y_gt = np.linalg.norm(y_gt)

        y_diff = y_gt - y_hat

        alpha = r * (1.0 - y_gt.dot(y_hat) / (n_y_gt * n_y_hat + eps)) ** 2
        beta = q * (y_diff.dot(y_diff)) / (n_y_gt * n_y_gt + eps)

        local_error = np.sqrt(alpha + beta)
        total_error += local_error

    average_error = total_error / float(N)

    return average_error


def n_choose_k_dataset_split(total_demos, test_num):
    """
    Generates all possible train-test splits of the dataset.

    Parameters:
    - total_demos (int): Total number of demonstrations in the dataset.
    - test_num (int): Number of demonstrations to be used for the test set.

    Returns:
    - (list, list): Two lists containing the train and test indices for each split.
    """
    assert 0 < test_num <= total_demos, "test_num must be between 1 and total_demos inclusive."

    arr = list(range(total_demos))
    comb = combinations(arr, test_num)

    train_indices = []
    test_indices = []

    arr_set = set(arr)
    for c in comb:
        c_set = set(c)
        rest = arr_set - c_set

        train_indices.append(list(rest))
        test_indices.append(list(c_set))

    return train_indices, test_indices

def train_test_val_split(seed, data, train_num=4, val_num=2, test_num=1):
    """
    Splits the data into training, validation, and test sets based on the specified numbers.

    Parameters:
    - seed (int): Random seed for reproducibility.
    - data (object): Dataset from which to extract the splits.
    - train_num (int): Number of samples in the training set.
    - val_num (int): Number of samples in the validation set.
    - test_num (int): Number of samples in the test set.

    Returns:
    - tuple: (x_train, y_train, x_val, y_val, x_test, y_test), the split data.
    """
    if train_num + val_num + test_num != 7:
        print("Split values must add up to 7")
        return -1

    np.random.seed(seed)
    indices = list(range(7))

    def get_split(idxs, num):
        selected_idxs = []
        if num == 0:
            return None, None, selected_idxs
        for _ in range(num):
            idn = np.random.choice(idxs)
            selected_idxs.append(idn)
            idxs.remove(idn)
        return lasa_to_numpy(data, selected_idxs), selected_idxs

    # Split data
    (x_train, y_train), train_indices = get_split(indices, train_num)
    (x_val, y_val), val_indices = get_split(indices, val_num)
    (x_test, y_test), test_indices = get_split(indices, test_num)

    # Print the indices for each split
    print("Train indices:", train_indices)
    print("Validation indices:", val_indices)
    print("Test indices:", test_indices)

    return x_train, y_train, x_val, y_val, x_test, y_test


def lasa_to_numpy(data, idns=None, smooth_val=5):
    """
    Converts LASA dataset demonstrations to numpy arrays of positions and velocities.

    Parameters:
    - data (object): The LASA dataset object containing demonstrations.
    - idns (list, optional): List of indices of demonstrations to include. Defaults to all (range 7).
    - smooth_val (int): Smoothing value applied to the position data.

    Returns:
    - tuple: (np_x, np_y), where np_x is the positions and np_y is the velocities.
    """
    if idns is None:
        idns = list(range(7))

    demos = data.demos
    demo_0 = demos[idns[0]]

    pos = np.array([smooth(demo_0.pos[0], smooth_val), smooth(demo_0.pos[1], smooth_val)])
    dt_global = demo_0.dt
    vel = (pos[:, 1:] - pos[:, :-1]) / dt_global
    pos = pos[:, 1:]

    for i in range(1, len(idns)):
        demo_i = demos[idns[i]]
        i_pos = np.array([smooth(demo_i.pos[0], smooth_val), smooth(demo_i.pos[1], smooth_val)])
        i_vel = (i_pos[:, 1:] - i_pos[:, :-1]) / dt_global
        i_pos = i_pos[:, 1:]

        pos = np.concatenate((pos, i_pos), axis=1)
        vel = np.concatenate((vel, i_vel), axis=1)

    np_x = pos.T
    np_y = vel.T

    return np_x, np_y



def decorate_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.gridn(axis='y', color="0.9", linestyle='-', linewidnth=1)
    ax.gridn(axis='x', color="0.9", linestyle='-', linewidnth=1)
    ax.set_axisbelow(True)
