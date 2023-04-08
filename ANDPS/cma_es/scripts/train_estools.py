from utils import *
from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
from es import CMAES, SimpleGA, OpenES, PEPG
import argparse
import time
import torch
import torch.nn as nn
import gym
from gym import spaces
import geotorch
import dartpy
import RobotDART as rd


np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))


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

displacements = [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.0],
                [-0.02, 0.3, 0.0],
                [-0.1, -0.2, 0.0],
                [0.05, -0.03, 0.0],
                [0.2, -0.5, 0.0],
]



MAX_STEPS = 1000
# ES related code
num_episode = 1
eval_steps = 25  # evaluate every N_eval steps
retrain_mode = True
cap_time_mode = True

num_worker = 8
num_worker_trial = 16

population = num_worker * num_worker_trial


optimizer = 'pepg'
antithetic = True
batch_mode = 'mean'

# seed for reproducibility
seed_start = 0

# name of the file (can override):
filebase = None

game = None
model = None
num_params = -1

es = None

# MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

PRECISION = 10000
SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
RESULT_PACKET_SIZE = 4*num_worker_trial
###


def count_parameters(model):
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return


class ActorAndps(nn.Module):
    def __init__(self, ds_dim, lims, N=4):
        super(ActorAndps, self).__init__()
        self.N = N
        # print("N = ", N)
        self.ds_dim = ds_dim
        self.n_params = ds_dim
        self.target_lims_x = torch.Tensor([])
        self.target_lims_y = torch.Tensor([])
        self.target_lims_z = torch.Tensor([])
        self.all_params_B_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.positive_semidefinite(self.all_params_B_A[i])

        self.all_params_C_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.skew(self.all_params_C_A[i])

        self.all_weights = nn.Sequential(
            nn.Linear(self.ds_dim + 3, 10), nn.ReLU(), nn.Linear(10, N), nn.Softmax(dim=1))
        # if lims is None:
        self.p = nn.Parameter(torch.randn(self.ds_dim))  # .to(device)
        self.lims = lims
        self.x_tar = nn.Parameter(torch.randn(self.ds_dim))
        self.env = PourEnv()
        self.param_count = self.count_parameters()
        # else:
        #     p = nn.Parameter(torch.randn(self.ds_dim))
        #     # print(lims[0])
        #     # print(lims[1])
        #     # input()
        #     self.x_tar = (torch.Tensor(lims[0]).requires_grad_(False) + torch.tanh(p) * torch.Tensor(lims[1]).requires_grad_(False)).to(device)

    def forward(self, x):
        # print("_"*10)
        # print(x)
        x_c = torch.Tensor(x[:6].reshape(-1, 6))
        # print(x_c)
        batch_size = x_c.shape[0]
        # print(batch_size)
        # print("_"*10)
        s_all = torch.zeros((1, self.ds_dim))
        w_all = self.all_weights(torch.Tensor(x.reshape(-1, 9)))

        # self.x_tar = (torch.Tensor(self.lims[0]).requires_grad_(
            # False) + torch.tanh(self.p) * torch.Tensor(self.lims[1]).requires_grad_(False))
        # print(self.x_tar)
        for i in range(self.N):
            A = (self.all_params_B_A[i].weight + self.all_params_C_A[i].weight)
            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(
                A, (self.x_tar-x_c).transpose(0, 1)).transpose(0, 1))
        return s_all

    def set_model_params(self, model_params):
        i = 0
        model_dict = self.state_dict()
        for key in model_dict.keys():
            # print(key, model_dict[key], model_dict[key].shape, model_dict[key].numel())
            # print(i, i+model_dict[key].numel())
            model_dict[key] = torch.Tensor(
                model_params[i:model_dict[key].numel() + i].reshape(model_dict[key].shape))
            i += model_dict[key].numel()
        # print(i)
        self.load_state_dict(model_dict)

    def count_parameters(self):
        count = 0
        model_dict = self.state_dict()
        for key in model_dict.keys():
            count += model_dict[key].numel()
        return count


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
        self.observation_space = gym.spaces.Box(low=np.array([np.float32(np.float32(-np.pi)), np.float32(-np.pi/2), np.float32(-np.pi), np.float32(self.low_bounds[0]), np.float32(
            self.low_bounds[1]), np.float32(self.low_bounds[2])]), high=np.array([np.pi, np.pi/2, np.pi, self.high_bounds[0], self.high_bounds[1], self.high_bounds[2]]), shape=(6,), dtype=np.float32)

    def step(self, action):
        self.simu.step_world()
        # print("_"*10)
        # print(action)
        # print("_"*10)
        observation = 0
        reward = 0
        done = False

        eulerXYZ = self.get_state()[:3]
        vel_rot = EREulerXYZ(eulerXYZ) @ action[0, :3]
        jac_pinv = damped_pseudoinverse(
            self.robot.jacobian(self.eef_link_name))
        # print("_"*10)
        # print(action)
        # print(vel_rot)
        # print(np.append(vel_rot,action[:3]))
        # print("_"*10)
        cmd = jac_pinv @ np.append(vel_rot, action[0, :3])
        self.robot.set_commands(cmd)
        self.simu.step_world()

        observation = self.get_state()
        reward = self.calc_reward()

        if (self.it == self.max_steps) or any(observation[3:6] > self.high_bounds) or any(observation[3:6] < self.low_bounds):
            done = True
            self.it = -1

        self.it += 1
        return observation, reward, done, {}

    def get_state(self):
        poseXYZ = self.robot.body_pose(self.eef_link_name).translation()
        eulerXYZ = dartpy.math.matrixToEulerXYZ(
            self.robot.body_pose(self.eef_link_name).matrix()[:3, :3])
        bowlXYZ = self.bowl.base_pose().translation()
        return np.append(np.append(eulerXYZ, poseXYZ), bowlXYZ)

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
        table_color = [0.933, 0.870, 0.784, 0.]
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
        # add some noise to the bowl position with a 50% chance
        # if np.random.random() > 0.5:
        idx = np.random.randint(0, 5)
        translation_noise = displacements[idx]
        # else:
        # translation_noise = np.zeros(3)
        tf.set_translation(self.table.base_pose().translation() +
                           [-0.3, 0.2, 0.37] + translation_noise)
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

        reward_rot = np.cos(angle_wrap(-np.pi/2 - dartpy.math.matrixToEulerXYZ(
            self.robot.body_pose("panda_ee").rotation())[2]))
        # print("Reward rot:", reward_rot)
        reward_pos = np.exp(-0.5 * np.linalg.norm(self.bowl.base_pose().translation()[
                            :2] - self.robot.body_pose("panda_ee").translation()[:2])/(p**2))
        for cereal in self.cereal_arr:
            reward += np.exp(-0.5 * np.linalg.norm(self.bowl.base_pose(
            ).translation() - cereal.base_pose().translation())/(p**2))
            # reward+= int(box_into_bowl(cereal.base_pose().translation(),self.bowl.base_pose().translation(), dartpy.math.matrixToEulerXYZ(self.bowl.base_pose().rotation())[2]))
        return 10 * reward + 10 * reward_rot + reward_pos

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

        A_eX = np.pi  # * (-1,1)
        A_eY = np.pi/2
        A_eZ = np.pi
        A_z = 1.190/2
        A_x = 0.855
        A_y = 0.855
        return np.array([[b_eX, b_eY, b_eZ, b_x, b_y, b_z], [A_eX, A_eY, A_eZ, A_x, A_y, A_z]]).copy()

    def viz_target(self, position):
        tf = dartpy.math.Isometry3()
        tf.set_translation(position)
        target_robot = rd.Robot.create_box([0.1, 0.1, 0.1], tf, "fixed", mass=0.1, color=[
                                           0, 1., 0., 1.], box_name="target")
        self.simu.add_robot(target_robot)


def initialize_settings(sigma_init=0.1, sigma_decay=0.9999):
    global population, filebase, game, model, num_params, es, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE
    population = num_worker * num_worker_trial
    os.makedirs('log', exist_ok=True)
    filebase = 'log/'+gamename+'.'+optimizer + \
        '.'+str(num_episode)+'.'+str(population)

    model = ActorAndps(6, PourEnv().get_limits())
    num_params = model.param_count
    print("size of model", num_params)

    if optimizer == 'ses':
        ses = PEPG(num_params,
                   sigma_init=sigma_init,
                   sigma_decay=sigma_decay,
                   sigma_alpha=0.2,
                   sigma_limit=0.02,
                   elite_ratio=0.1,
                   weight_decay=0.005,
                   popsize=population)
        es = ses
    elif optimizer == 'ga':
        ga = SimpleGA(num_params,
                      sigma_init=sigma_init,
                      sigma_decay=sigma_decay,
                      sigma_limit=0.02,
                      elite_ratio=0.1,
                      weight_decay=0.005,
                      popsize=population)
        es = ga
    elif optimizer == 'cma':
        cma = CMAES(num_params,
                    sigma_init=sigma_init,
                    popsize=population)
        es = cma
    elif optimizer == 'pepg':
        pepg = PEPG(num_params,
                    sigma_init=sigma_init,
                    sigma_decay=sigma_decay,
                    sigma_alpha=0.20,
                    sigma_limit=0.02,
                    learning_rate=0.01,
                    learning_rate_decay=1.0,
                    learning_rate_limit=0.01,
                    weight_decay=0.005,
                    popsize=population)
        es = pepg
    else:
        oes = OpenES(num_params,
                     sigma_init=sigma_init,
                     sigma_decay=sigma_decay,
                     sigma_limit=0.02,
                     learning_rate=0.01,
                     learning_rate_decay=1.0,
                     learning_rate_limit=0.01,
                     antithetic=antithetic,
                     weight_decay=0.005,
                     popsize=population)
        es = oes

    PRECISION = 10000
    SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
    RESULT_PACKET_SIZE = 4*num_worker_trial
###


def sprint(*args):
    print(args)  # if python3, can do print(*args)
    sys.stdout.flush()


class OldSeeder:
    def __init__(self, init_seed=0):
        self._seed = init_seed

    def next_seed(self):
        result = self._seed
        self._seed += 1
        return result

    def next_batch(self, batch_size):
        result = np.arange(self._seed, self._seed+batch_size).tolist()
        self._seed += batch_size
        return result


class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2**31-1)

    def next_seed(self):
        result = np.random.randint(self.limit)
        return result

    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result


def encode_solution_packets(seeds, solutions, train_mode=1, max_len=-1):
    n = len(seeds)
    result = []
    worker_num = 0
    for i in range(n):
        worker_num = int(i / num_worker_trial) + 1
        result.append([worker_num, i, seeds[i], train_mode, max_len])
        result.append(np.round(np.array(solutions[i])*PRECISION, 0))
    result = np.concatenate(result).astype(np.int32)
    result = np.split(result, num_worker)
    return result


def decode_solution_packet(packet):
    packets = np.split(packet, num_worker_trial)
    result = []
    for p in packets:
        result.append([p[0], p[1], p[2], p[3], p[4],
                      p[5:].astype(float)/PRECISION])
    return result


def encode_result_packet(results):
    r = np.array(results)
    r[:, 2:4] *= PRECISION
    return r.flatten().astype(np.int32)


def decode_result_packet(packet):
    r = packet.reshape(num_worker_trial, 4)
    workers = r[:, 0].tolist()
    jobs = r[:, 1].tolist()
    fits = r[:, 2].astype(float)/PRECISION
    fits = fits.tolist()
    times = r[:, 3].astype(float)/PRECISION
    times = times.tolist()
    result = []
    n = len(jobs)
    for i in range(n):
        result.append([workers[i], jobs[i], fits[i], times[i]])
    return result


def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):

    reward_list = []
    t_list = []
    RENDER_DELAY = render_mode

    max_episode_length = 3000

    if train_mode and max_len > 0:
        if max_len < max_episode_length:
            max_episode_length = max_len

    for episode in range(num_episode):

        obs = model.env.reset()

        if obs is None:
            obs = np.zeros(model.ds_dim)

        total_reward = 0.0
        reward_threshold = 0  # consider we have won if we got more than this

        for t in range(max_episode_length):

            if render_mode:
                model.env.render()
                if RENDER_DELAY:
                    time.sleep(0.01)

            action = model.forward(obs).detach().numpy()

            prev_obs = obs

            #noise = np.random.randn(len(action))
            #action += noise

            obs, reward, done, info = model.env.step(action)

            if (render_mode):
                pass
                #print("action", action, "step reward", reward)
                #print("step reward", reward)
            total_reward += reward

        if render_mode:
            print("reward", total_reward, "timesteps", t)
        reward_list.append(total_reward)
        t_list.append(t)

    return reward_list, t_list


def worker(weights, seed, train_mode_int=1, max_len=-1):

    train_mode = (train_mode_int == 1)
    model.set_model_params(weights)
    reward_list, t_list = simulate(
        model, train_mode=train_mode, render_mode=False, num_episode=num_episode, seed=seed, max_len=max_len)
    if batch_mode == 'min':
        reward = np.min(reward_list)
    else:
        reward = np.mean(reward_list)
    t = np.mean(t_list)
    return reward, t


def slave():
    # model.make_env()
    packet = np.empty(SOLUTION_PACKET_SIZE, dtype=np.int32)
    while 1:
        comm.Recv(packet, source=0)
        assert (len(packet) == SOLUTION_PACKET_SIZE)
        solutions = decode_solution_packet(packet)
        results = []
        for solution in solutions:
            worker_id, jobidx, seed, train_mode, max_len, weights = solution
            assert (train_mode == 1 or train_mode == 0), str(train_mode)
            worker_id = int(worker_id)
            possible_error = "work_id = " + \
                str(worker_id) + " rank = " + str(rank)
            assert worker_id == rank, possible_error
            jobidx = int(jobidx)
            seed = int(seed)
            fitness, timesteps = worker(weights, seed, train_mode, max_len)
            results.append([worker_id, jobidx, fitness, timesteps])
        result_packet = encode_result_packet(results)
        assert len(result_packet) == RESULT_PACKET_SIZE
        comm.Send(result_packet, dest=0)


def send_packets_to_slaves(packet_list):
    num_worker = comm.Get_size()
    assert len(packet_list) == num_worker-1
    for i in range(1, num_worker):
        packet = packet_list[i-1]
        assert (len(packet) == SOLUTION_PACKET_SIZE)
        comm.Send(packet, dest=i)


def receive_packets_from_slaves():
    result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

    reward_list_total = np.zeros((population, 2))

    check_results = np.ones(population, dtype=int)
    for i in range(1, num_worker+1):
        comm.Recv(result_packet, source=i)
        results = decode_result_packet(result_packet)
        for result in results:
            worker_id = int(result[0])
            possible_error = "work_id = " + \
                str(worker_id) + " source = " + str(i)
            assert worker_id == i, possible_error
            idx = int(result[1])
            reward_list_total[idx, 0] = result[2]
            reward_list_total[idx, 1] = result[3]
            check_results[idx] = 0

    check_sum = check_results.sum()
    assert check_sum == 0, check_sum
    return reward_list_total


def evaluate_batch(model_params, max_len=-1):
    # duplicate model_params
    solutions = []
    for i in range(es.popsize):
        solutions.append(np.copy(model_params))

    seeds = np.arange(es.popsize)

    packet_list = encode_solution_packets(
        seeds, solutions, train_mode=0, max_len=max_len)

    send_packets_to_slaves(packet_list)
    reward_list_total = receive_packets_from_slaves()

    reward_list = reward_list_total[:, 0]  # get rewards
    return np.mean(reward_list)


def master():

    start_time = int(time.time())
    sprint("training", gamename)
    sprint("population", es.popsize)
    sprint("num_worker", num_worker)
    sprint("num_worker_trial", num_worker_trial)
    sys.stdout.flush()

    seeder = Seeder(seed_start)

    filename = filebase+'.json'
    filename_log = filebase+'.log.json'
    filename_hist = filebase+'.hist.json'
    filename_best = filebase+'.best.json'

    # model.make_env()

    t = 0

    history = []
    eval_log = []
    best_reward_eval = 0
    best_model_params_eval = None

    max_len = -1  # max time steps (-1 means ignore)

    while True:
        t += 1

        solutions = es.ask()

        if antithetic:
            seeds = seeder.next_batch(int(es.popsize/2))
            seeds = seeds+seeds
        else:
            seeds = seeder.next_batch(es.popsize)

        packet_list = encode_solution_packets(
            seeds, solutions, max_len=max_len)

        send_packets_to_slaves(packet_list)
        reward_list_total = receive_packets_from_slaves()

        reward_list = reward_list_total[:, 0]  # get rewards

        mean_time_step = int(
            np.mean(reward_list_total[:, 1])*100)/100.  # get average time step
        max_time_step = int(
            np.max(reward_list_total[:, 1])*100)/100.  # get average time step
        avg_reward = int(np.mean(reward_list)*100) / \
            100.  # get average time step
        std_reward = int(np.std(reward_list)*100)/100.  # get average time step

        es.tell(reward_list)

        es_solution = es.result()
        model_params = es_solution[0]  # best historical solution
        reward = es_solution[1]  # best reward
        curr_reward = es_solution[2]  # best of the current batch
        model.set_model_params(np.array(model_params).round(4))

        r_max = int(np.max(reward_list)*100)/100.
        r_min = int(np.min(reward_list)*100)/100.

        curr_time = int(time.time()) - start_time

        h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(
            es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)

        if cap_time_mode:
            max_len = 2*int(mean_time_step+1.0)
        else:
            max_len = -1

        history.append(h)

        with open(filename, 'wt') as out:
            res = json.dump([np.array(es.current_param()).round(
                4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

        with open(filename_hist, 'wt') as out:
            res = json.dump(history, out, sort_keys=False,
                            indent=0, separators=(',', ':'))

        sprint(gamename, h)

        if (t == 1):
            best_reward_eval = avg_reward
        if (t % eval_steps == 0):  # evaluate on actual task at hand

            prev_best_reward_eval = best_reward_eval
            model_params_quantized = np.array(es.current_param()).round(4)
            reward_eval = evaluate_batch(model_params_quantized, max_len=-1)
            model_params_quantized = model_params_quantized.tolist()
            improvement = reward_eval - best_reward_eval
            eval_log.append([t, reward_eval, model_params_quantized])
            with open(filename_log, 'wt') as out:
                res = json.dump(eval_log, out)
            if (len(eval_log) == 1 or reward_eval > best_reward_eval):
                best_reward_eval = reward_eval
                best_model_params_eval = model_params_quantized
            else:
                if retrain_mode:
                    sprint(
                        "reset to previous best params, where best_reward_eval =", best_reward_eval)
                    es.set_mu(best_model_params_eval)
            with open(filename_best, 'wt') as out:
                res = json.dump([best_model_params_eval, best_reward_eval],
                                out, sort_keys=True, indent=0, separators=(',', ': '))
            sprint("improvement", t, improvement, "curr", reward_eval,
                   "prev", prev_best_reward_eval, "best", best_reward_eval)
            sprint("Learned target: ", model.x_tar.data.numpy())


def main(args):
    global gamename, optimizer, num_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, cap_time_mode
    gamename = "Pour"
    optimizer = args.optimizer
    num_episode = args.num_episode
    eval_steps = args.eval_steps
    num_worker = args.num_worker
    num_worker_trial = args.num_worker_trial
    antithetic = (args.antithetic == 1)
    retrain_mode = (args.retrain == 1)
    cap_time_mode = (args.cap_time == 1)
    seed_start = args.seed_start

    initialize_settings(args.sigma_init, args.sigma_decay)

    sprint("process", rank, "out of total ", comm.Get_size(), "started")
    if (rank == 0):
        master()
    else:
        slave()


def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        print(["mpirun", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(
            ["mpirun", "-np", str(n), sys.executable] + ['-u'] + sys.argv, env=env)
        return "parent"
    else:
        global nworkers, rank
        nworkers = comm.Get_size()
        rank = comm.Get_rank()
        print('assigning the rank and nworkers', nworkers, rank)
        return "child"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using pepg, ses, openes, ga, cma'))
    # parser.add_argument('gamename', type=str,
    # help='robo_pendulum, robo_ant, robo_humanoid, etc.')
    parser.add_argument('-o', '--optimizer', type=str,
                        help='ses, pepg, openes, ga, cma.', default='cma')
    parser.add_argument('-e', '--num_episode', type=int,
                        default=1, help='num episodes per trial')
    parser.add_argument('--eval_steps', type=int, default=10,
                        help='evaluate every eval_steps step')
    parser.add_argument('-n', '--num_worker', type=int, default=8)
    parser.add_argument('-t', '--num_worker_trial', type=int,
                        help='trials per worker', default=4)
    parser.add_argument('--antithetic', type=int, default=1,
                        help='set to 0 to disable antithetic sampling')
    parser.add_argument('--cap_time', type=int, default=0,
                        help='set to 0 to disable capping timesteps to 2x of average.')
    parser.add_argument('--retrain', type=int, default=0,
                        help='set to 0 to disable retraining every eval_steps if results suck.\n only works w/ ses, openes, pepg.')
    parser.add_argument('-s', '--seed_start', type=int,
                        default=111, help='initial seed')
    parser.add_argument('--sigma_init', type=float,
                        default=0.40, help='sigma_init')
    parser.add_argument('--sigma_decay', type=float,
                        default=0.999, help='sigma_decay')

    args = parser.parse_args()
    if "parent" == mpi_fork(args.num_worker+1):
        os.exit()
    main(args)
