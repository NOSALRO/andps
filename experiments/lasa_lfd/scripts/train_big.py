import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from andps import ANDP, CustomDataset
from utils import simple_nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_folder = os.path.join('data', 'lasa')

# Names of demos to be used from the Lasa handwriting dataset
names = ["JShape", "Angle", "Khamesh", "LShape", "PShape", "RShape",
         "Sharpc", "Sine", "Spoon", "Trapezoid", "Worm", "WShape"]

# "Ideal" number of dynamical systems needed for every shape
num_DSs = [6, 3, 4, 6, 7, 6, 5, 5, 6, 6, 5, 6]

# Number of dataset splits (every demo is split into 35 combinations)
N_trains = 35

experiment = 'andps'
# experiment = 'simple_nn'

for name, num_DS in zip(names, num_DSs):
    for k in range(N_trains):
        data_lasa = np.load(os.path.join(dataset_folder, f'{name}_{k}.npz'))

        X_train = data_lasa['X_train']
        Y_train = data_lasa['Y_train']

        dim = X_train.shape[1]
        target = X_train[-1]

        model = simple_nn(dim) if experiment == 'simple_nn' else ANDP(
            dim, num_DS, torch.Tensor(target), device=device)
        model.to(device)

        batch_size = 64
        batches_per_epoch = X_train.shape[0] // batch_size
        epochs = 1000
        learning_rate = 1e-3
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.1)

        dataset = CustomDataset(X_train.copy(), Y_train.copy())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_train_loss = np.inf

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, outputs in dataloader:
                inputs, outputs = inputs.to(
                    device).view(-1, dim), outputs.to(device).view(-1, dim)
                optimizer.zero_grad()
                output = model(inputs)
                loss = torch.nn.functional.mse_loss(
                    output, outputs, reduction='mean')
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_mean_loss = running_loss / batches_per_epoch
            if train_mean_loss < best_train_loss:
                best_train_loss = train_mean_loss
                torch.save(model.state_dict(), os.path.join(
                    'models', experiment, f'lasa_{name}_{k}_best.pt'))

            print(
                f"{experiment} {name}({num_DS}, {k}) -> Epoch: {epoch+1} Loss: {train_mean_loss:.6f}")

        print('Finished Training')
        torch.save(model.state_dict(), os.path.join(
            'models', f'lasa_{name}_{k}_{epoch+1}.pt'))
