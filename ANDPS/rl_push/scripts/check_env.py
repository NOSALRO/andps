from stable_baselines3.common.env_checker import check_env
from env import PushEnv

env = PushEnv(enable_graphics=False, enable_record=False, seed=0)

check_env(env, warn=True)

obs = env.reset()
n_steps = 100
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print("Reward:", reward, "Done:", done)
    if done:
        obs = env.reset()