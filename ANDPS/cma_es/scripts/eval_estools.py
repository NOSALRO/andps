import numpy as np
import RobotDART as rd
from envs import PourEnv
import torch
import torch.nn as nn
import geotorch
import json

class ActorAndps(nn.Module):
    def __init__(self, ds_dim, lims, N=4):
        super(ActorAndps, self).__init__()
        self.N = N
        # print("N = ", N)
        self.ds_dim = ds_dim
        self.n_params = ds_dim
        self.target_lims_x = torch.Tensor([])
        self.target_lims_y = torch.Tensor([])
        self.target_lims_z = torch.Tensor([])
        self.all_params_B_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.positive_semidefinite(self.all_params_B_A[i])

        self.all_params_C_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.skew(self.all_params_C_A[i])

        self.all_weights = nn.Sequential(
            nn.Linear(self.ds_dim + 3, 10), nn.ReLU(), nn.Linear(10, N), nn.Softmax(dim=1))
        # if lims is None:
        self.p = nn.Parameter(torch.randn(self.ds_dim))  # .to(device)
        self.lims = lims
        self.x_tar = nn.Parameter(torch.randn(self.ds_dim))
        self.env = PourEnv()
        self.param_count = self.count_parameters()
        # else:
        #     p = nn.Parameter(torch.randn(self.ds_dim))
        #     # print(lims[0])
        #     # print(lims[1])
        #     # input()
        #     self.x_tar = (torch.Tensor(lims[0]).requires_grad_(False) + torch.tanh(p) * torch.Tensor(lims[1]).requires_grad_(False)).to(device)

    def forward(self, x):
        # print("_"*10)
        # print(x)
        x_c = torch.Tensor(x[:6].reshape(-1, 6))
        # print(x_c)
        batch_size = x_c.shape[0]
        # print(batch_size)
        # print("_"*10)
        s_all = torch.zeros((1, self.ds_dim))
        w_all = self.all_weights(torch.Tensor(x.reshape(-1, 9)))

        # self.x_tar = (torch.Tensor(self.lims[0]).requires_grad_(
            # False) + torch.tanh(self.p) * torch.Tensor(self.lims[1]).requires_grad_(False))
        # print(self.x_tar)
        for i in range(self.N):
            A = (self.all_params_B_A[i].weight + self.all_params_C_A[i].weight)
            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(
                A, (self.x_tar-x_c).transpose(0, 1)).transpose(0, 1))
        return s_all

    def set_model_params(self, model_params):
        i = 0
        model_dict = self.state_dict()
        for key in model_dict.keys():
            # print(key, model_dict[key], model_dict[key].shape, model_dict[key].numel())
            # print(i, i+model_dict[key].numel())
            model_dict[key] = torch.Tensor(
                model_params[i:model_dict[key].numel() + i].reshape(model_dict[key].shape))
            i += model_dict[key].numel()
        # print(i)
        self.load_state_dict(model_dict)

    def count_parameters(self):
        count = 0
        model_dict = self.state_dict()
        for key in model_dict.keys():
            count += model_dict[key].numel()
        return count


# read json file
path = 'log/1.json'
json_file = open(path, 'r')
params = json.load(json_file)[0]


env = PourEnv(enable_graphics=True, enable_record=True, seed=-1, dt=0.01)
# define andps model
model = ActorAndps(6, env.get_limits())
model.set_model_params(np.array(params))
print(model.x_tar)
model.forward(np.array([0, 0, 0, 0, 0, 0,0,0,0]))
print(model.x_tar)
input
# env = PourEnv(enable_graphics=True, enable_record=True, seed=-1, dt=0.01)
env.viz_target(model.x_tar[3:].detach().numpy())

state = env.reset()
done = False
while not done:
    action = model(state)
    state, reward, done, _ = env.step(action.detach().numpy())
    # state = env.get_state()
    # print(state.shape)
    # print(model.x_tar)
# for _ in range(1000):
#     env.robot.set_commands([-1], ['panda_joint4'])
#     env.simu.step_world()

#     env.calc_reward()