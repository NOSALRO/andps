import sys
import os
from scipy.io import savemat
import numpy as np
from utils import n_choose_k_dataset_split, lasa_to_numpy
import pyLasaDataset as lasa

datas = [lasa.DataSet.JShape, lasa.DataSet.Angle, lasa.DataSet.Khamesh, lasa.DataSet.LShape, lasa.DataSet.PShape, lasa.DataSet.RShape, lasa.DataSet.Sharpc, lasa.DataSet.Sine, lasa.DataSet.Spoon, lasa.DataSet.Trapezoid, lasa.DataSet.Worm, lasa.DataSet.WShape]

names = ["JShape", "Angle", "Khamesh", "LShape", "PShape", "RShape", "Sharpc", "Sine", "Spoon", "Trapezoid", "Worm", "WShape"]

if __name__ == "__main__":
    # args = sys.argv[1:]
    n_test = 3
    dataset_folder = os.path.join('data', 'lasa')
    if len(sys.argv[1:]) > 0:
        n_test = int(sys.argv[1])
    if len(sys.argv[1:]) > 1:
        dataset_folder = sys.argv[2]

    for i in range(len(names)):
        name = names[i]
        data = datas[i]
        train_idxs, test_idxs = n_choose_k_dataset_split(len(data.demos), n_test)

        for k in range(len(train_idxs)):
            X_train, Y_train = lasa_to_numpy(data, train_idxs[k])
            X_test, Y_test = lasa_to_numpy(data, test_idxs[k])

            # save for python/C++
            np.savez(os.path.join(dataset_folder, name + '_' + str(k) + '.npz'), X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

            # save for matlab
            mdic = {'X_train' : X_train, 'Y_train' : Y_train, 'X_test' : X_test, 'Y_test' : Y_test}
            savemat(os.path.join(dataset_folder, name + '_' + str(k) + '.mat'), mdic)
