from env import PushEnv

from stable_baselines3 import SAC as algo
from stable_baselines3.common.noise import NormalActionNoise
from utils import SACDensePolicy
import numpy as np

EPOCHS = 1000


push_env = PushEnv(enable_graphics=False,
                   enable_record=False, seed=-1, dt=0.01)
push_env.reset()

model = algo(SACDensePolicy, push_env, verbose=1, learning_rate=0.001)
# model = algo("MlpPolicy", env, verbose=1, learning_rate=5e-4, train_freq=MAX_STEPS, gradient_steps=200, batch_size=256, learning_starts=256, action_noise=NormalActionNoise(0., 1.))
# model = algo.load("cereal_killer")
# model.set_env(env)
# model.learning_rate = 5e-4
model.learn(total_timesteps=400 * EPOCHS)
model.save("cereal_killer")


obs = push_env.reset()
for i in range(push_env.max_steps):
    action, _state = model.predict(obs, deterministic=True)
    print(action, _state)
    obs, reward, done, info = push_env.step(action.reshape(3,))
    # if i == 0:
    # env.render()
    if done:
        break

print("Done")
