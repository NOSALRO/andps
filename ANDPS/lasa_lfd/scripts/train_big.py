import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import pyLasaDataset as lasa
from utils import net_first_order as net
from utils import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_folder = os.path.join('data', 'lasa')

names = ["JShape", "Angle", "Khamesh", "LShape", "PShape", "RShape", "Sharpc", "Sine", "Spoon", "Trapezoid", "Worm", "WShape"]
num_DSs = [6, 3, 4, 6, 7, 6, 5, 5, 6, 6, 5, 6]

N_trains = 35

for i in range(len(names)):
    name = names[i]
    num_DS = num_DSs[i]
    for k in range(N_trains):
        data_lasa = np.load(os.path.join(dataset_folder, name + '_' + str(k) + '.npz'))

        X_train = data_lasa['X_train']
        Y_train = data_lasa['Y_train']

        X_test = data_lasa['X_test']
        Y_test = data_lasa['Y_test']

        dim = X_train.shape[1]

        # end of dataset creation
        target = X_train[-1]

        model = net(dim, num_DS, target, device)
        model.to(device)

        batch_size = 64
        dataset = CustomDataset(X_train.copy(), Y_train.copy())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batches_per_epoch = X_train.shape[0]//batch_size
        epochs = 1000
        le_r = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=le_r, weight_decay=0.1)

        best_train_loss = np.inf

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, output]
                inputs, outputs = data
                # print(inputs)
                batch_loss = 0.0
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output, _, _ = model(inputs.to(device).view(-1, dim), False)
                # print(output.shape, outputs.view(dim, -1).shape)
                # + 0.005 * (1-w_all).square().sum().sum()
                loss = torch.nn.functional.mse_loss(output, outputs.to(device).view(-1, dim), reduction='mean')
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                # scheduler.step(loss.item())
            # scheduler.step()
            train_mean_loss = running_loss/batches_per_epoch

            if train_mean_loss < best_train_loss:
                best_train_loss = train_mean_loss
                torch.save(model.state_dict(),'pytorch models/lasa_' + name + '_' + str(k) + '_best.pt')

            print(name + "(" + str(num_DS) + ", " + str(k) +") -> Epoch: ", epoch+1, "Loss: ", train_mean_loss)
        print('Finished Training')

        torch.save(model.state_dict(),'pytorch models/lasa_' + name + '_' + str(k) + '_' + str(epoch + 1) + '.pt')
