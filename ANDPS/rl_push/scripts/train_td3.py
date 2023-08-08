import numpy as np
import torch
import gym
import argparse
import os
import matplotlib.pyplot as plt
import utils
import td3
from env import PushEnv

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(policy, env_name, seed, eval_episodes=2):
	eval_env = PushEnv(enable_graphics=True, enable_record=False, seed=seed+100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # OpenAI gym environment name
    parser.add_argument("--env", default="PushEnv")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=5e3, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.1, type=float)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99,
                        type=float)     # Discount factor
    # Target network update rate
    parser.add_argument("--tau", default=0.005, type=float)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.2)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="")
    args = parser.parse_args()

    file_name = f"td3_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: td3, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
       os.makedirs("./models")

    env = PushEnv(enable_graphics=False, enable_record=False, seed=args.seed)

	# Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau}

    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq

    policy = td3.TD3(**kwargs)


    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env.max_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if ((t + 1) % args.eval_freq == 0) and (t >= args.start_timesteps):
            evaluations.append(eval_policy(policy, args.env, args.seed))
            # Plot evaluation results
            # plt.plot(evaluations)
            # plt.savefig(f"./plots/{file_name}.png")
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")