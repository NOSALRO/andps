from env import PushEnv

from stable_baselines3 import SAC as algo

import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 10000
MAX_STEPS = 400

push_env = PushEnv(enable_graphics=True,
                   enable_record=True, seed=-1, dt=0.01)

model = algo.load("models/push_trained")


obs = push_env.reset()
cumu_reward = 0
for i in range(push_env.max_steps):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = push_env.step(action.reshape(3,))

    cumu_reward +=reward
    print(reward)

    # if i == 0:
    # env.render()
    if done:
        break

print("Done Reward: ", cumu_reward)
