# Sensorimotor Learning with Stability Guarantees via Autonomous Neural Dynamic Policies

Code for the paper entitled "Sensorimotor Learning with Stability Guarantees via Autonomous Neural Dynamic Policies" by *Dionis Totsila\*, Konstantinos Chatzilygeroudis\*, Valerio Modugno, Denis Hadjivelichkov, and Dimitrios Kanoulas*

\* Equal contribution

![Screenshot from 2024-01-10 15-45-20](https://github.com/NOSALRO/andps/assets/50770773/78e57186-d31e-46d6-8027-3f3df28a995e)

## Abstract

State-of-the-art sensorimotor learning algorithms, either in the context of reinforcement learning or imitation learning, offer policies that can often produce unstable behaviors, damaging the robot and/or the environment. Moreover, it is very difficult to interpret the optimized controller and analyze its behavior and/or performance. Traditional robot learning, on the contrary, relies on dynamical system-based policies that can be analyzed for stability/safety. Such policies, however, are neither flexible nor generic and usually work only with proprioceptive sensor states. In this work, we bridge the gap between generic neural network policies and dynamical system-based policies, and we introduce Autonomous Neural Dynamic Policies (ANDPs) that: (a) are based on autonomous dynamical systems, (b) always produce asymptotically stable behaviors, and (c) are more flexible than traditional stable dynamical system-based policies. ANDPs are fully differentiable, flexible generic-policies that can be used for both imitation learning and reinforcement learning setups, while ensuring asymptotic stability. Through several experiments, we explore the flexibility and capacity of ANDPs in several imitation learning tasks including experiments with image observations. The results show that ANDPs combine the benefits of both neural network-based and dynamical system-based methods.

You can find more information about the method in the [video](https://youtu.be/ZI9-TLSovpQ).

## Maintainers

- Dionis Totsila (INRIA) - dionis.totsila@inria.fr
- Konstantinos Chatzilygeroudis (Univ. of Patras) - costashatz@upatras.gr

## Citing ANDPs

If you use this code in a scientific publication, please use the following citation:

```bibtex
@article{dionis2024andps,
        title={{Sensorimotor Learning with Stability Guarantees via Autonomous Neural Dynamic Policies}},
        author={Totsila, Dionis and Chatzilygeroudis, Konstantinos and Modugno, Valerio and Hadjivelichkov, Denis and Kanoulas, Dimitrios},
        year={2024},
        journal={{Preprint}}
      }
```

## How to use the library

First install andps either from pip:
```bash
pip install -i https://test.pypi.org/simple/ andps==0.1.0
```
or from source:
```bash
git clone https://github.com/NOSALRO/andps/tree/main
cd andps
python -m build .
pip install -e .
```

Then you can use the ANDPs in your code as follows:
```python
from andps import ANDP

ds_dim = 3 # dimension of the dynamical system
N = 2 # number of Dynamical Systems
attractor = torch.Tensor([0.0, 0.0, 0.0]) # attractor point
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ANDP(ds_dim, N, attractor, device).to(device)
```
The default architecture of the state dependent weight function is:
```python
torch.nn.Sequential(torch.nn.Linear(ds_dim, 10), torch.nn.ReLU(), torch.nn.Linear(10, N), nn.Softmax(dim=1))
```
*For a more detailed example you can check [experiment 1](https://github.com/NOSALRO/andps/tree/lib/experiments/lasa_lfd)*

For systems that the full state contains only the controllable part, you can also define the hidden layers of the state dependent weight function as follows:
```python
from andps import ANDP

ds_dim = 3 # dimension of the dynamical system
N = 2 # number of Dynamical Systems
attractor = torch.Tensor([0.0, 0.0, 0.0]) # attractor point
hidden_layers = [128, 64, 32]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ANDP(ds_dim, N, attractor, hidden_layers=hidden_layers, device=device).to(device)

```

The above example creates the following architecture for the state dependent weight function:
```python
torch.nn.Sequential(nn.Linear(self.ds_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4), nn.Softmax(dim=1))
```
*For a more detailed example you can check [experiment 3](https://github.com/NOSALRO/andps/tree/lib/experiments/pouring_task_lfd)*


If our state vector also contains the non-controllable part of the system, we can define a custom state dependent weight function as follows, keep in mind that we also need a new forward method because the default forward method assumes that the state vector contains only the controllable part of the system:
```python
import torch
from torch import nn
from torch.nn import functional as F
from andps import ANDP

# For this experiment we need to define a custom Weighting function
# Which means that we need to override the forward as well

# First we define the custom weighting function
# we decided to define the state as the controllable part of the state and then the image
# so our current control state is x[:3] and the image is x[3:]
class CNN_weight(nn.Module):
    def __init__(self, ds_dim, N):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(1690+ds_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        self.fc6 = nn.Linear(64, N)

    def forward(self, state):
        # this expects the images as well as the current state
        x = self.pool1(
            F.relu(self.conv1(state[:, 3:].reshape(state.shape[0], 1, 64, 64))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.cat((x, state[:, :3]), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc6(x), dim=1)

        return x


# let's define a class that inherits from ANDP
class cnn_ANDP(ANDP):
    def __init__(self, ds_dim, N, target, custom_weighting_function, device):
        super().__init__(ds_dim, N, target, all_weights=custom_weighting_function, device=device)


    def forward(self, x):
        batch_size = x.shape[0]
        s_all = torch.zeros((1, self.ds_dim)).to(x.device)
        w_all = self.all_weights(x)

        for i in range(self.N):
            A = self.all_params_B_A[i].weight + self.all_params_C_A[i].weight
            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(
                A, (self.x_target-x[:, :3]).transpose(0, 1)).transpose(0, 1))
        return s_all


ds_dim = 3 # dimension of the dynamical system
N = 2 # number of Dynamical Systems
attractor = torch.Tensor([0.0, 0.0, 0.0]) # attractor point
hidden_layers = [128, 64, 32]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cnn_ANDP(ds_dim, N, attractor, CNN_weight(dim, num_DS), device=device).to(device)
```


*For a more detailed example you can check [experiment 2](https://github.com/NOSALRO/andps/tree/lib/experiments/panda_images)*

## Acknowledgments

Konstantinos Chatzilygeroudis was supported by the [Hellenic Foundation for Research and Innovation](https://www.elidek.gr/en/homepage/) (H.F.R.I.) under the "3rd Call for H.F.R.I. Research Projects to support Post-Doctoral Researchers" (Project Acronym: [NOSALRO](https://nosalro.github.io/), Project Number: 7541). Dimitrios Kanoulas and Valerio Modugno were supported by the UKRI Future Leaders Fellowship [MR/V025333/1] (RoboHike).

<p align="center">
<img src="https://www.elidek.gr/wp-content/themes/elidek/images/elidek_logo_en.png" alt="logo_elidek" width="40%"/><br/>
<img src="https://www.cinuk.org/content/uploads/2022/11/UKRI-logo2.png" alt="logo_ukri" width="40%"/>
<p/>

This work was conducted as collaboration of the [Computational Intelligence Lab](http://cilab.math.upatras.gr/) (CILab), Department of Mathematics, University of Patras, Greece, and the [Robot Perception and Learning Lab](https://rpl-as-ucl.github.io/) (RPL Lab), Department of Computer
Science, University College London (UCL), United Kingdom.

<p align="center">
<img src="https://nosalro.github.io/images/logo_cilab.jpg" alt="logo_cilab" width="50%"/>
<img src="https://www.upatras.gr/wp-content/uploads/up_2017_logo_en.png" alt="logo_cilab" width="50%"/>
<img src="https://rpl-as-ucl.github.io/images/logos/rpl-cs-ucl-logo.png" alt="logo_rpl" width="50%"/>
</p>

## License

[BSD 2-Clause &#34;Simplified&#34; License](https://opensource.org/license/bsd-2-clause/)
