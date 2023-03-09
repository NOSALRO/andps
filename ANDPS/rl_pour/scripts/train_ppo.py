import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool

from utils_ppo import PPO

import dartpy

from utils import damped_pseudoinverse, angle_wrap
import numpy as np
import RobotDART as rd

MAX_STEPS = 500

def EREulerXYZ(eulerXYZ):
    x = eulerXYZ[0]
    y = eulerXYZ[1]
    # z = eulerXYZ[2]
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)

    R = np.zeros((3, 3))
    R[0, 0] = 1.
    R[0, 2] = sin_y
    R[1, 1] = cos_x
    R[1, 2] = -cos_y * sin_x
    R[2, 1] = sin_x
    R[2, 2] = cos_x * cos_y

    return R


class PourEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_graphics=True, enable_record=True, seed=-1, dt=0.01):
        self.setup_simu(dt)
        if (enable_graphics):
            self.setup_graphics(enable_record)
        self.setup_env()

        # cartesianVel, eulerXYZ
        # self.state = np.array([[0.,0.,0.,0.,0.,0.]])
        self.seed = seed
        self.it = 0
        self.max_steps = MAX_STEPS

        # define action space
        self.action_space = gym.spaces.Box(low=np.array(
            [-1, -1, -1, -1, -1, -1]), high=np.array([1, 1, 1, 1, 1, 1]), shape=(6,), dtype=np.float32)
        self.low_bounds = [self.table.base_pose().translation(
        )[0]-1.5, self.table.base_pose().translation()[1]-1., self.table.base_pose().translation()[2]+0.35]
        self.high_bounds = [self.table.base_pose().translation()[0]+1.5, self.table.base_pose(
        ).translation()[1]+1., self.robot.body_pose("panda_ee").translation()[2]+1.]
        self.observation_space = gym.spaces.Box(low=np.array([np.float32(np.float32(-np.pi)), np.float32(-np.pi/2), np.float32(-np.pi), np.float32(self.low_bounds[0]), np.float32(self.low_bounds[1]), np.float32(self.low_bounds[2])]), high=np.array([np.pi, np.pi/2, np.pi, self.high_bounds[0], self.high_bounds[1], self.high_bounds[2]]), shape=(6,), dtype=np.float32)

    def step(self, action):
        self.simu.step_world()
        observation = 0
        reward = 0
        done = False

        eulerXYZ = self.get_state()[:3]
        vel_rot = EREulerXYZ(eulerXYZ) @ action[:3]
        jac_pinv = damped_pseudoinverse(
            self.robot.jacobian(self.eef_link_name))
        # print("_"*10)
        # print(action)
        # print(vel_rot)
        # print(np.append(vel_rot,action[:3]))
        # print("_"*10)
        cmd = jac_pinv @ np.append(vel_rot, action[:3])
        self.robot.set_commands(cmd)
        self.simu.step_world()

        observation = self.get_state()
        reward = self.calc_reward()

        if (self.it == self.max_steps) or any(observation[3:] > self.high_bounds) or any(observation[3:] < self.low_bounds):
            done = True
            self.it = -1

        self.it += 1
        return observation, reward, done, {}

    def get_state(self):
        poseXYZ = self.robot.body_pose(self.eef_link_name).translation()
        eulerXYZ = dartpy.math.matrixToEulerXYZ(
            self.robot.body_pose(self.eef_link_name).matrix()[:3, :3])
        return np.append(eulerXYZ, poseXYZ)

    def render(self):
        print(self.get_state())

    def setup_simu(self, dt):
        self.dt = dt
        # init simu
        self.simu = rd.RobotDARTSimu(self.dt)
        self.simu.set_collision_detector("fcl")
        # add checkerboard floor
        self.simu.add_checkerboard_floor()

    def setup_graphics(self, enable_record):
        self.gconfig = rd.gui.Graphics.default_configuration()
        self.gconfig.width = 640
        self.gconfig.height = 480
        self.graphics = rd.gui.Graphics(self.gconfig)
        self.simu.set_graphics(self.graphics)
        self.graphics.look_at((1.5, 1.5, 2.), (0.0, 0.0, 0.5))
        if (enable_record):
            self.graphics.camera().record(True)
            self.graphics.record_video(
                "cerial-env.mp4", self.simu.graphics_freq())

    def setup_table(self):
        # table_packages = [("table", "urdfs/table")]
        # self.table = rd.Robot("urdfs/table/table.urdf",   tabie_packages, "table")
        # self.table.set_color_mode("material")
        table_dims = [3., 2., 0.7]
        table_pose = [0, 0, 0, 0, 0, 0.35]
        table_color = [0.933, 0.870, 0.784, 1.]
        self.table = rd.Robot.create_box(
            table_dims, table_pose, "fix", mass=30., color=table_color, box_name="table")
        self.table.fix_to_world()
        self.simu.add_robot(self.table)

    def setup_robot(self):
        franka_packages = [
            ("franka_description", "urdfs/franka/franka_description")]
        self.robot = rd.Robot("urdfs/franka/franka.urdf",
                              franka_packages, "franka")
        self.robot.set_color_mode("material")
        self.simu.add_robot(self.robot)
        self.eef_link_name = "panda_ee"
        tf = self.robot.base_pose()
        tf.set_translation([-0.7, 0, 0.7])
        self.robot.set_base_pose(tf)
        self.robot.fix_to_world()
        self.robot.set_position_enforced(True)
        self.robot.set_actuator_types("servo")
        self.reset_robot()

    def reset_robot(self):
        self.robot.reset()
        positions = self.robot.positions()
        positions[4] = np.pi / 2.0
        positions[5] = np.pi / 2.0
        positions[6] = 3*np.pi/4
        # positions[7] = 0.015
        # positions[8] = 0.015
        self.robot.set_positions(positions)

    # def setup_cereal_box(self):
    #     cereal_packages = [("cereal", "urdfs/cereal")]
    #     self.cereal_box = rd.Robot(
    #         "urdfs/cereal/cereal.urdf",  cereal_packages, "cereal_box")
    #     self.cereal_box.set_color_mode("material")
    #     self.simu.add_robot(self.cereal_box)
    #     self.reset_cereal_box()

    # def reset_cereal_box(self):
    #     tf = dartpy.math.Isometry3()
    #     tf.set_translation(self.robot.body_pose(
    #         self.eef_link_name).translation() + [0.0, 0.065, 0.05])
    #     self.cereal_box.set_base_pose(tf)

    def setup_bowl(self):
        bowl_packages = [("bowl", "urdfs/bowl")]
        self.bowl = rd.Robot("urdfs/bowl/bowl.urdf",  bowl_packages, "bowl")
        self.bowl.set_color_mode("material")
        self.simu.add_robot(self.bowl)
        self.reset_bowl()

    def reset_bowl(self):
        tf = self.bowl.base_pose()
        tf.set_translation(
            self.table.base_pose().translation() + [-0.3, 0.2, 0.37])
        self.bowl.set_base_pose(tf)
        self.bowl.fix_to_world()

    def add_cereal(self, count=4):
        box_tf = self.robot.body_pose("panda_ee")
        self.cereal_arr = []

        cereal_dims = [0.02, 0.02, 0.02]
        cereal_mass = 1
        cereal_color = [1, 0., 0., 1.]

        for i in range(count):
            cereal_pose = [0., 0., 0., box_tf.translation()[0], box_tf.translation(
            )[1] - 0.06 + (i % 2)*0.01, box_tf.translation()[2] + (i % (2+1)) * 0.001]
            # cereal = rd.Robot.create_ellipsoid(cereal_dims, cereal_pose, "free", mass=cereal_mass, color=cereal_color, ellipsoid_name="cereal_" + str(i))
            cereal = rd.Robot.create_box(
                cereal_dims, cereal_pose, "free", mass=cereal_mass, color=cereal_color, box_name="cereal " + str(i))
            self.cereal_arr.append(cereal)
            self.simu.add_robot(cereal)

    def reset_cereal(self):
        box_tf = self.robot.body_pose("cereal_box")
        for i in range(len(self.cereal_arr)):
            cereal_pose = [0., 0., 0., box_tf.translation()[0], box_tf.translation()[
                1]-0.05, box_tf.translation()[2] + i * 0.015]
            self.cereal_arr[i].reset()
            self.cereal_arr[i].set_base_pose(cereal_pose)

    def setup_env(self):
        print("Initializing Rd Environment")
        self.setup_table()
        print("Added table")
        self.setup_robot()
        print("Added robot")
        # self.setup_cereal_box()
        # print("Added cereal box")
        self.add_cereal()
        print("Added cereal")
        self.setup_bowl()
        print("Added bowl")

    def reset(self):
        self.reset_robot()
        # self.reset_cereal_box()
        self.reset_cereal()
        self.reset_bowl()
        # print("Env reset successfully")
        for _ in range(100):
            self.simu.step_world()
        return self.get_state()

    def calc_reward(self):
        reward = 0
        p = 0.2
        for cereal in self.cereal_arr:
            reward+= np.exp(-0.5 * np.linalg.norm(self.bowl.base_pose().translation() -cereal.base_pose().translation())/(p**2))

        return reward

    def get_limits(self):

        # lim_eX = [-np.pi, np.pi]
        # lim_eY = [-np.pi/2, np.pi/2]
        # lim_eZ = [-np.pi, np.pi]
        # lim_z = [self.robot.base_pose().translation()[2], self.robot.base_pose().translation()[0] + 1.190]
        # lim_x = [self.robot.base_pose().translation()[0]-0.855, self.robot.base_pose().translation()[1] + 0.855]
        # lim_y = [self.robot.base_pose().translation()[1]-0.855, self.robot.base_pose().translation()[2] + 0.855]

        # I want to write the lims in the form y= Ax + b
        # x has to be 1 value and not a tuble, and as a result i want the limits to be symmetrical so:

        b_eX = 0
        b_eY = 0
        b_eZ = 0
        b_z = self.robot.base_pose().translation()[2] + 1.190/2
        b_x = self.robot.base_pose().translation()[0]
        b_y = self.robot.base_pose().translation()[1]

        A_eX = np.pi # * (-1,1)
        A_eY = np.pi/2
        A_eZ = np.pi
        A_z = 1.190/2
        A_x = 0.855
        A_y = 0.855
        return np.array([[b_eX, b_eY, b_eZ, b_x, b_y,b_z ], [A_eX, A_eY, A_eZ, A_x, A_y, A_z]]).copy()





################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "PourEnv-v2"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = MAX_STEPS              # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 2        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.8                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len //2      # update policy every n timesteps
    K_epochs = 256               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 5e-4       # learning rate for actor network
    lr_critic = 1e-3       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = PourEnv(True, True,seed =random_seed)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################
    lims = env.get_limits()
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, lims)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
