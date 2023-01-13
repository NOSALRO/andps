# from cmath import inf
# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import  DataLoader
# import pyLasaDataset as lasa
# from utils import simple_nn as net_a, compute_error as err
# from utils import CustomDataset
# import matplotlib.pyplot as plt
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset_folder = os.path.join('data', 'lasa/')
# dt_global = 0.00432448
# names = ["Angle", "JShape", "Khamesh", "LShape", "PShape", "RShape", "Sharpc", "Sine", "Spoon", "Trapezoid"]#, "Worm"]
# num_DSs = [3,6,4,6,7,6,5,5,6,6]#,5]
# w_xmin = inf
# w_xmax = -inf
# w_ymin = inf
# w_ymax = -inf

# N_trains = 34
# for ikr in range(len(names)):
#     name = names[ikr]
#     num_DS = num_DSs[ikr]
#     # seds_dataset_folder = os.path.join('data', 'SEDS', name)
#     for k in range(N_trains):
#         print(name)
#         data_lasa = np.load(os.path.join(dataset_folder + name + '_' + str(k) + '.npz'))
#         X_train = data_lasa['X_train']
#         Y_train = data_lasa['Y_train']

#         X_test = data_lasa['X_test']
#         Y_test = data_lasa['Y_test']

#         net = net_a(2)#,num_DS,[0,0])

#         net.load_state_dict(torch.load('pytorch models/simple_nn/lasa_'+names[ikr]+'_'+str(k)+'_1000.pt',map_location='cpu'))

#         with torch.no_grad():
#             Y_model = net(torch.Tensor(X_train).view(-1,2)).detach().cpu()

#             train_error = err(X_train, Y_train, Y_model.numpy())
#             Y_model_test = net(torch.Tensor(X_test).view(-1,2)).detach().cpu()

#             test_error = err(X_test, Y_test, Y_model_test.numpy())

#             cur_error = np.array([train_error, test_error])
#             if k == 0:
#                 all_errors = cur_error
#             else:
#                 all_errors = np.vstack([all_errors, cur_error])
#             # print(k)
#     print(all_errors)
#     print(all_errors.shape)
#     print(all_errors[:,0])
#     np.savez('neurips/data/errors/nn/'+ names[ikr] + '_errors.npz', train_error =all_errors[:,0], test_error=all_errors[:,1])
