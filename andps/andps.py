
import torch
from torch import nn
import geotorch


class ANDP(nn.Module):
    """The Autonomous Neural Dynamic Policy (ANDP) pytorch model"""

    def __init__(self, ds_dim: int, N: int, attractor: torch.Tensor, all_weights: nn.Module = None, hidden_layers: list = [10], device='cpu'):
        '''
            Initializes the Autonomous Neural Dynamic Policy (ANDP) model

            Args:
            ds_dim (int): Dimensions of the Dynamical System(s)
            N (int): Number of Dynamical Systems
            attractor (torch.Tensor): Fixed attractor point(s) for the dynamical system(s) (must be of shape ds_dim)
            all_weights (nn.Module, optional): The weighting functions for the dynamical systems must me a nn.Module
            hidden_layers (list, optional): The hidden layers for the default weighting function. Defaults to [10].
            device (str, optional): Device to run the model on. Defaults to 'cpu'.
        '''
        super().__init__()
        self.N = N
        self.ds_dim = ds_dim

        # In order to satisfy the stability guarantees, we need to ensure that the Ai + Ai^T > 0
        # To handle this in an unconstrained manner, we parameterize the matrices A as B + C, where B is symmetric positive definite and C is skew-symmetric

        # Initialize the B matrices
        self.all_params_B_A = nn.ModuleList(
            [nn.Linear(self.ds_dim, self.ds_dim, bias=False) for i in range(N)])
        # Ensure that the B matrices are symmetric positive definite
        for i in range(N):
            geotorch.positive_semidefinite(self.all_params_B_A[i])

        # Initialize the C matrices
        self.all_params_C_A = nn.ModuleList(
            [nn.Linear(self.ds_dim, self.ds_dim, bias=False) for i in range(N)])
        # Ensure that the C matrices are skew-symmetric
        for i in range(N):
            geotorch.skew(self.all_params_C_A[i])

        # State-dependent weight function
        if all_weights is None:

            # let's define the weighting function using the hidden layers array
            self.all_weights = nn.Sequential()
            self.all_weights.add_module(
                '0', nn.Linear(self.ds_dim, hidden_layers[0]))
            self.all_weights.add_module('1', nn.ReLU())

            for i in range(1, len(hidden_layers)):
                self.all_weights.add_module(
                    f'{i+2}', nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.all_weights.add_module(f'{i+3}', nn.ReLU())

            self.all_weights.add_module(
                f'{len(hidden_layers)+1}', nn.Linear(hidden_layers[-1], N))
            self.all_weights.add_module('softmax', nn.Softmax(dim=1))

        else:
            # wee need to assert that the input of the weighting function is the same as the state dimension
            assert all_weights[0].in_features == self.ds_dim, "The input of the first layer of the weighting function must be equal to the state dimension"
            # we need to assert that the output of the weighting function is the same as the number of dynamical systems
            assert all_weights[-1].out_features == self.N, "The output of the last layer of the weighting function must be equal to the number of dynamical systems"

            self.all_weights = all_weights

        self.x_target = attractor.view(-1, self.ds_dim).to(device)

    def forward(self, x_cur):

        assert x_cur.shape[1] == self.ds_dim, "The input state dimension must be equal to the state dimension of the dynamical system"

        batch_size = x_cur.shape[0]

        # initialize the sum of the non-linear combination of the linear dynamical systems
        out = torch.zeros((1, self.ds_dim)).to(x_cur.device)

        w_all = self.all_weights(x_cur)

        for i in range(self.N):
            # construct the matrix A_i = B_i + C_i
            A = (self.all_params_B_A[i].weight + self.all_params_C_A[i].weight)
            # batch friendly computation
            out = out + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(
                A, (self.x_target-x_cur).transpose(0, 1)).transpose(0, 1))
        return out
