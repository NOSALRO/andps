import torch
from torch.utils.data import Dataset

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

