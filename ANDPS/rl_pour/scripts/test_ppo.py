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

def box_into_basket(box_translation, basket_translation, basket_angle):
    basket_xy_corners = np.array([basket_translation[0] + 0.14, basket_translation[0] + 0.14, basket_translation[0] - 0.14, basket_translation[0] - 0.14,
                                  basket_translation[1] - 0.08, basket_translation[1] + 0.08, basket_translation[1] + 0.08, basket_translation[1] - 0.08], dtype=np.float64).reshape(2, 4)

    rotation_matrix = np.array([np.cos(basket_angle), np.sin(basket_angle), -np.sin(basket_angle), np.cos(basket_angle)], dtype=np.float64).reshape(2, 2)

    basket_center = np.array([basket_translation[0], basket_translation[1]], dtype=np.float64).reshape(2, 1)
    rotated_basket_xy_corners = np.matmul(rotation_matrix, (basket_xy_corners - basket_center)) + basket_center

    d1 = (rotated_basket_xy_corners[0][1] - rotated_basket_xy_corners[0][0]) * (box_translation[1] - rotated_basket_xy_corners[1][0]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][0]) * (rotated_basket_xy_corners[1][1] - rotated_basket_xy_corners[1][0])
    d2 = (rotated_basket_xy_corners[0][2] - rotated_basket_xy_corners[0][1]) * (box_translation[1] - rotated_basket_xy_corners[1][1]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][1]) * (rotated_basket_xy_corners[1][2] - rotated_basket_xy_corners[1][1])
    d3 = (rotated_basket_xy_corners[0][3] - rotated_basket_xy_corners[0][2]) * (box_translation[1] - rotated_basket_xy_corners[1][2]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][2]) * (rotated_basket_xy_corners[1][3] - rotated_basket_xy_corners[1][2])
    d4 = (rotated_basket_xy_corners[0][0] - rotated_basket_xy_corners[0][3]) * (box_translation[1] - rotated_basket_xy_corners[1][3]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][3]) * (rotated_basket_xy_corners[1][0] - rotated_basket_xy_corners[1][3])

    if ((d1 > 0.0) and (d2 > 0.0) and (d3 > 0.0) and (d4 > 0.0) and (box_translation[2] <= 0.04)):
        return True
    else:
        return False


class PourEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_graphics=False, enable_record=False, seed=-1, dt=0.01):
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
        reward_cereal = 0
        p = 0.2
        for cereal in self.cereal_arr:
            reward_cereal+= np.exp(-0.5 * np.linalg.norm(self.bowl.base_pose().translation() -cereal.base_pose().translation())/(p**2)) / len(self.cereal_arr)
            # reward_cereal+= int(box_into_basket(cereal.base_pose().translation(),self.bowl.base_pose().translation(), dartpy.math.matrixToEulerXYZ(self.bowl.base_pose().rotation())[2]))
        reward_rot = np.cos(angle_wrap(-np.pi/2 - dartpy.math.matrixToEulerXYZ(self.robot.body_pose(self.eef_link_name).matrix()[:3, :3])[2]))
        reward_pos = np.linalg.norm(self.bowl.base_pose().translation()[:2]-self.robot.body_pose(self.eef_link_name).translation()[:2])
        w0 = 10.
        w1 = 1.
        w2 = 5.
        return w0*reward_cereal + w1*reward_rot + w2*reward_pos

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





#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "PourEnv-v3"
    has_continuous_action_space = True
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 1    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = PourEnv(True,True)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    lims = env.get_limits()
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, lims)


    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()