import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import net_first_order as net_a, decorate_axis, simple_nn as net_b
from utils import count_parameters
from palettable.scientific.diverging import Vik_5
from numpy import genfromtxt, vstack


lasa_dataset_folder = os.path.join('data', 'lasa')


# Datasets used for the paper
names = ["JShape", "Khamesh", "Sine", "Spoon", "Trapezoid"]

# Corresponding number of dynamical systems per dataset
lasa_id = [6, 4, 5, 6, 6]
num_DSs = [6, 6, 5, 5, 5]
# Which split id was used for every dataset
ks = [10, 25, 7, 5, 24]
experiment = 'andps'
# experiment = 'simple_nn'

dt_global = 0.00432448

for id in range(0, len(names)):
    k = ks[id]
    print(names[id], k)
    name = names[id]
    num_DS = num_DSs[id]
    # we need this for nice axis limits
    seds_dataset_folder = os.path.join('data', 'SEDS', name)
    path = seds_dataset_folder + '/' + name + '_'+str(k)+'_DS_'+str(lasa_id[id])

    w_x = genfromtxt(path + '_YLIM.csv', delimiter=',')
    w_y = genfromtxt(path + '_XLIM.csv', delimiter=',')
    w_xmin = w_x[0]
    w_xmax = w_x[1]
    w_ymin = w_y[0]
    w_ymax = w_y[1]
    data_lasa = np.load(os.path.join(
        lasa_dataset_folder, name + '_' + str(k) + '.npz'))
    X_train = data_lasa['X_train']
    Y_train = data_lasa['Y_train']

    path = seds_dataset_folder + '/' + name + '_'+str(k)+'_DS_'+str(num_DS)
    X_test = data_lasa['X_test']
    Y_test = data_lasa['Y_test']
    dataset = np.vstack([X_train, X_test])

    dim = X_train.shape[1]

    # end of dataset creation
    target = X_train[-1]
    if experiment == 'andps':
        net = net_a(dim, num_DS,[0,0])
    else:
        net = net_b(dim)

    net.load_state_dict(torch.load('models/'+experiment+'/lasa_' + name +'_'+str(k) + '_1000.pt', map_location='cpu'))
    net.eval()

    # network evaluation

    start_idxs = []
    end_idxs = []
    for i in range(0, X_train.shape[0], 999):
        start_idxs.append(i)
        end_idxs.append(i+999)
    ###print(X_train, X_train.shape)

    # print(start_idxs)

    initial_positions = []

    original_start_idx = start_idxs[0]
    # get  starting point
    x_cur_0 = X_train[original_start_idx, :2].copy()
    trajectory = x_cur_0.copy()
    print("Simulation")
    print("Train set")
    for idk in range(0, len(start_idxs)):
        print("Starting point: ", idk)
        start_idx = start_idxs[idk]
        end_idx = end_idxs[idk]
        x_cur_0 = X_train[start_idx, :2].copy()
        # get and plot starting point
        # start Evaluation
        for _ in range(2000):
            # ,False)[0].detach().cpu()
            v = net(torch.Tensor(
                (np.array([x_cur_0[0], x_cur_0[1]]))).view(-1, 2)).detach().cpu()
            x_cur_0[0] = x_cur_0[0] + dt_global * v[0, 0].numpy()
            x_cur_0[1] = x_cur_0[1] + dt_global * v[0, 1].numpy()
            if (x_cur_0[0] > w_xmax):
                break
            if (x_cur_0[1] > w_ymax):
                break
            if (x_cur_0[0] < w_xmin):
                break
            if (x_cur_0[1] < w_ymin):
                break
            trajectory = np.vstack([trajectory, x_cur_0[0:2]])

    start_idxs = []
    end_idxs = []
    for i in range(0, X_test.shape[0], 999):
        start_idxs.append(i)
        end_idxs.append(i+999)

    initial_positions = []

    original_start_idx = start_idxs[0]
    # get  starting point
    x_cur_0 = X_test[original_start_idx, :2].copy()
    test_trajectory = x_cur_0.copy()
    print("Test set")
    for idk in range(0, len(start_idxs)):
        print("Starting point: ", idk)
        start_idx = start_idxs[idk]
        end_idx = end_idxs[idk]
        x_cur_0 = X_test[start_idx, :2].copy()
        # get and plot starting point

        # start Evaluation
        for _ in range(2000):
            v = net(torch.Tensor(
                (np.array([x_cur_0[0], x_cur_0[1]]))).view(-1, 2)).detach().cpu()
            x_cur_0[0] = x_cur_0[0] + dt_global * v[0, 0].numpy()
            x_cur_0[1] = x_cur_0[1] + dt_global * v[0, 1].numpy()
            if (x_cur_0[0] > w_xmax):
                break
            if (x_cur_0[1] > w_ymax):
                break
            if (x_cur_0[0] < w_xmin):
                break
            if (x_cur_0[1] < w_ymin):
                break
            test_trajectory = np.vstack([test_trajectory, x_cur_0[0:2]])

    Y, X = np.mgrid[w_ymin:w_ymax:200j, w_xmin:w_xmax:200j]
    U = X.copy()
    V = Y.copy()

    # Calculate streamlines
    total_size = 200*200
    N_cuts = 20
    cut_step = int(total_size / N_cuts)
    X_all = X.copy().reshape((-1, total_size))
    Y_all = Y.copy().reshape((-1, total_size))
    D_in = np.concatenate((X_all, Y_all)).T

    v_all = np.zeros((total_size, dim))

    for kr in range(N_cuts):
        p = torch.Tensor(D_in[kr*cut_step:(kr+1)*cut_step, :]).view(-1, 2)
        v = net(p)

        v_all[kr*cut_step:(kr+1)*cut_step, :] = v.detach().cpu().numpy()

    U = v_all[:, 0].reshape((200, 200))
    V = v_all[:, 1].reshape((200, 200))

    # plot streamlines
    x_init = dataset[0, :]
    for i in range(1, 7):
        x_init = vstack([x_init, dataset[i*999, :]])

    # get target
    x_tar = X_train[end_idx-1, :2].copy()

    # plt.show()
    np.savez('plots/data/'+experiment+'/'+names[id]+'_'+str(k)+'.npz', X=X, Y=Y, U=U, V=V, dataset_pos=dataset,
             train_trajectory=trajectory, test_trajectory=test_trajectory, initial_pos=x_init, target_pos=x_tar)
