import numpy as np
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from TD3 import TD3, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set target
target = np.array([[1., 1.]])

class env():
    def __init__(self, seed = -1, dt = 0.01):
        self.state = np.array([[0., 0.]])
        self.seed = seed
        self.dt = dt
        self.it = 0
        self.max_steps = 500

    def reset(self):
        self.it = 0
        self.state = np.array([[0., 0.]])
        return self.state

    def step(self, action):
        self.it += 1
        self.state = self.state + action.copy() * self.dt
        observation = self.state.copy()
        dist = np.inner(observation - target, observation - target)[0]
        p = 1.
        reward = np.exp(-0.5*dist/(p*p))[0]
        done = False
        # if(dist < 0.1) :
        #     reward = 1000
        if (self.it == self.max_steps) or (abs(self.state[0][0]) > 4. or abs(self.state[0][1]) > 4.):
            self.it == -1
            done = True
        return observation, reward, done

def eval_policy(policy, env, eval_episodes=10):
    env.reset()
    if env.seed > 0:
        set_seeds(env.seed+10)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        count = 0
        while not done:
            action = policy.select_action(np.array(state))
            action = action.reshape(-1, 3)
            state, reward, done, count = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def set_seeds(seed):
    if seed < 0:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


env = env()
state_dim = 2
action_dim = 2
actor_type = 'andps'

# policy = TD3(state_dim, action_dim, policy_noise=0.001, lr_actor=1e-3, lr_critic=1e-3)
policy = TD3(state_dim, action_dim, actor_type = actor_type, discount=0.999, policy_noise=0.1, noise_clip=0.05, policy_freq=2, lr_actor=1e-4, lr_critic=1e-3)

replay_buffer = ReplayBuffer(state_dim, action_dim)

# Evaluate untrained policy
#eval_policy(policy, robot, simu, seed, 10)
evaluations = []
eval_freq = 10e3
state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0
max_env_episodes = 2e3
start_timesteps = env.max_steps * 10 # 10 random episodes
batch_size = 256

trajectory = np.array(np.zeros([1, 2]))
prev_action = None
same_action_prob = 0.999
for t in range(int(1e6)):
    episode_timesteps += 1
    # Select action randomly or according to policy
    if t < start_timesteps:
        if (prev_action is not None) and np.random.randint(0, 100) < (100 * same_action_prob):
            action = prev_action.copy()
            same_action_prob *= same_action_prob
        else:
            action = np.random.normal(0, 2, [1, 2])
            prev_action = action
            same_action_prob = 0.999
        # print("random action", action)
        # print(action)
    else:
        # print(action)
        action = (policy.select_action(np.array(state)) + np.random.normal(0., 0.1, size=action_dim)).reshape(-1, 2)

    # Perform action
    next_state, reward, done = env.step(action)
    # print(next_state)

    done_bool = float(done) if episode_timesteps < max_env_episodes else 0

    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)

    state = next_state
    episode_reward += reward

    # # Train agent after collecting sufficient data
    # if t >= start_timesteps:
    #     # print("TRAINING PHASE")
    #     # for _ in range(env.max_steps):
    #     policy.train(replay_buffer, batch_size)

    trajectory = np.vstack([trajectory, state.copy()])


    if done:
        # Plot episode's trajectory
        fig, ax = plt.subplots()
        fig.suptitle("Episode Num: " + str(episode_num + 1) + " Reward: " + str(episode_reward))
        ax.scatter(trajectory[:,0], trajectory[:,1], c = 'slateblue', label='trajectory')
        ax.scatter(0, 0, c = 'r', marker="o", label='start')
        ax.scatter(1, 1, c = 'g', marker="*", label='goal')
        if actor_type == 'andps':
            ax.scatter(policy.actor.x_tar[0].detach().cpu().numpy(), policy.actor.x_tar[1].detach().cpu().numpy(), c = 'b', marker="^", label='learned goal')
        plt.legend(markerscale = .5)
        plt.savefig("plots/" + str(episode_num) + ".png")

        plt.clf()
        plt.close()
        trajectory = np.zeros([1, 2])
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        state, done = env.reset(), False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        prev_action = None
        same_action_prob = 0.999

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            # print("TRAINING PHASE")
            for _ in range(env.max_steps):
                policy.train(replay_buffer, batch_size)

    # # Evaluate episode
    # if (t + 1) % eval_freq == 0:
    #     evaluations.append(eval_policy(policy, robot, simu, seed))
    #     print(evaluations)