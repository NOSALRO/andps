#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import pyLasaDataset as lasa

from torchvision import transforms
from PIL import Image
from utils import andps_images as net
device = "cpu"
convert_tensor = transforms.ToTensor()

# w vars needed for streamline plot


# Lasa dataset demos
data = [lasa.DataSet.JShape, lasa.DataSet.Angle, lasa.DataSet.Khamesh]

names = ["JShape", "Angle", "Khamesh"]

# read axis limits from csv file
limits = np.genfromtxt('data/lims.csv', delimiter=',')[1:, 1:]

# Images corresponding to each demo
# [convert_tensor(Image.open("data/train/Sine_demo_1.png")), convert_tensor(Image.open("data/train/Spoon_demo_1.png")), convert_tensor(Image.open("data/train/JShape_demo_1.png"))]
images = []


#  Create Dataset with image indexing
for d in range(len(data)):
    for i in range(7):
        images.append(convert_tensor(Image.open(
            "data/images/train/" + names[d] + "_demo_" + str(i+1) + ".png")))
        cur_demo = data[d].demos[i]
        cur_pos = cur_demo.pos
        cur_vel = cur_demo.vel
        dt_global = cur_demo.dt
        cur_np_X = cur_pos.transpose()
        temp = torch.Tensor([d*7 + i]).repeat(cur_np_X.shape[0], 1)
        cur_np_X = torch.cat((torch.Tensor(cur_np_X), temp), 1)
        cur_np_Y = torch.Tensor(cur_vel.transpose())
        if d == 0 and i == 0:
            np_X = cur_np_X
            np_Y = cur_np_Y
        else:
            np_X = torch.cat((np_X, cur_np_X), 0)
            np_Y = torch.cat((np_Y, cur_np_Y), 0)


start_idxs = np.zeros([3, 7])
end_idxs = np.zeros([3, 7])
for i in range(0, 3):
    for j in range(0, 7):
        start_idxs[i, j] = (i*7000+j*1000)
        end_idxs[i, j] = start_idxs[i, j]+1000

dim = np_Y.shape[1]


# all trajectories have the same target (0, 0)
target = np_X[-1, :2]

# Number of dynamical systems
num_DS = 9

net = net(dim, num_DS, target, len(names), images)
net.load_state_dict(torch.load(
    'models/fo_lasa_image_3_20.pt', map_location=torch.device('cpu')))
net.eval()

id = 0
for d in range(len(data)):
    initial_positions = []

    demo = data[d].demos[0]
    pos = demo.pos[:, 0].copy()
    vel = demo.vel[:, 0].copy()

    with torch.no_grad():

        original_start_idx = int(start_idxs[d, 0])
        original_end_idx = int(end_idxs[d, -1])
        x_cur_0 = np_X[original_start_idx, :2].numpy().copy()
        trajectory = x_cur_0[0:2].copy()
        for j in range(start_idxs.shape[1]):
            print("Evaluating demo: "+ names[d]+" "+str(d+1) +"/3 , Trajectory: " +str(j+1)+"/7")
            start_idx = int(start_idxs[d, j])
            x_cur_0 = np_X[start_idx, :2].numpy().copy()
            for epoch in range(2000):
                v = net(torch.Tensor((np.array([x_cur_0[0], x_cur_0[1], id]))).to(
                    device).view(-1, 3)).detach().cpu()
                x_cur_0[0] = x_cur_0[0] + dt_global * v[0, 0].numpy()
                x_cur_0[1] = x_cur_0[1] + dt_global * v[0, 1].numpy()
                trajectory = np.vstack([trajectory, x_cur_0[0:2]])

                if (np.linalg.norm(abs(x_cur_0 - target.numpy())) < 1e-1):
                    print("reached target on iteration: ", str(epoch))
                    break
            id += 1
        w_xmin = limits[d, 0]
        w_xmax = limits[d, 1]
        w_ymin = limits[d, 2]
        w_ymax = limits[d, 3]
        Y, X = np.mgrid[w_ymin:w_ymax:50j, w_xmin:w_xmax:50j]
        U = X.copy()
        V = Y.copy()

        # Calculate streamlines
        total_size = 50*50
        N_cuts = 10
        cut_step = int(total_size / N_cuts)
        X_all = X.copy().reshape((-1, total_size))
        Y_all = Y.copy().reshape((-1, total_size))
        id_all = np.ones((1, total_size)) * id-1
        D_in = np.concatenate((X_all, Y_all, id_all)).T
        v_all = np.zeros((total_size, dim))

        for k in range(N_cuts):
            p = torch.Tensor(
                D_in[k*cut_step:(k+1)*cut_step, :]).to(device).view(-1, 3)
            v = net(p)
            v_all[k*cut_step:(k+1)*cut_step, :] = v.detach().cpu().numpy()
            U = v_all[:, 0].reshape((50, 50))
            V = v_all[:, 1].reshape((50, 50))


        init = np_X[int(start_idxs[d, 0]), :2].numpy()
        for jk in range(1, start_idxs.shape[1]):
            init = np.vstack([init, np_X[int(start_idxs[d, jk]), :2]])


        x_tar = np_X[original_end_idx-1, :2].numpy().copy()
        # save data
        np.savez("data/lasa_image_data_" + names[d] +".npz", X = X, Y = Y, U = U, V = V, trajectory = trajectory, init = init, x_tar = x_tar, demo = np_X[original_start_idx:original_end_idx, :])

