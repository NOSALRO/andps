from env import TiagoEnv
import numpy as np

tiago_env = TiagoEnv(enable_graphics=True)

tiago_env.reset()

for _  in range(5000):
    tiago_env.step(np.array([0.1, 0.1, 0.1]))