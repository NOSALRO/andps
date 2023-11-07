from env import TiagoEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import SAC as algo
from stable_baselines3.common.noise import NormalActionNoise
from utils_sb3 import SACDensePolicy
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 10000
MAX_STEPS = 500

tiago_env = TiagoEnv(enable_graphics=False, seed=-
                     1, dt=0.01, max_steps=MAX_STEPS)
tiago_env.reset()

tiago_env = Monitor(tiago_env)
model = algo(SACDensePolicy, tiago_env, verbose=0, learning_rate=5e-4, train_freq=1,
             gradient_steps=-1, batch_size=2048, action_noise=NormalActionNoise(0., 1.))

# model = algo("MlpPolicy", env, verbose=1
# model = algo.load("cereal_killer")
# model.set_env(env)
# model.learning_rate = 5e-4

for i in range(EPOCHS):
    model.learn(total_timesteps=4000)
    model.save("models/tiago_sac_"+str((i+1)*4000))
    # episode_rewards = np.array(tiago_env.get_episode_rewards())



# Eval
obs = tiago_env.reset()
for i in range(tiago_env.max_steps):
    action, _state = model.predict(obs, deterministic=True)
    print(action, _state)
    obs, reward, done, info = tiago_env.step(action.reshape(3,))
    # if i == 0:
    # env.render()
    if done:
        break

print("Done")
