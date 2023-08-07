from env import PushEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import SAC as algo
from stable_baselines3.common.noise import NormalActionNoise
from utils import SACDensePolicy
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 10000
MAX_STEPS = 400

push_env = PushEnv(enable_graphics=False,
                   enable_record=False, seed=-1, dt=0.01, max_steps=MAX_STEPS)
push_env.reset()

push_env = Monitor(push_env)
model = algo(SACDensePolicy, push_env, verbose=0, learning_rate=1e-3, train_freq=1,
             gradient_steps=-1, batch_size=2048, action_noise=NormalActionNoise(0., 1.))

# model = algo("MlpPolicy", env, verbose=1
# model = algo.load("cereal_killer")
# model.set_env(env)
# model.learning_rate = 5e-4

for i in range(EPOCHS):
    model.learn(total_timesteps=4000)
    model.save("models/push_trained")

    # Retrieve the episode rewards from the monitor
    episode_rewards = np.array(push_env.get_episode_rewards())

    # Plot the learning curve
    plt.plot(np.arange(1, len(episode_rewards)+1), episode_rewards)
    # plt.hlines(0, 0, len(episode_rewards)+1, linestyles='dashed')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.title('SAC Learning Curve')
    plt.savefig("plots/sac_learning_curve.png")
    plt.close()


# Eval
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
