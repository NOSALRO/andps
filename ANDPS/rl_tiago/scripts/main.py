import gym
import dartpy
from stable_baselines3 import SAC as algo
from stable_baselines3.common.noise import NormalActionNoise
from utils_sb3 import SACDensePolicy
import numpy as np
import RobotDART as rd
import matplotlib.pyplot as plt
import scipy.stats

MAX_STEPS = 1600

class TiagoEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_graphics=False, seed=-1, dt=0.01):
        super(TiagoEnv, self).__init__()
        self.enable_graphics = enable_graphics

        # x, y, theta
        self.state = np.array([[0., 0.]])

        self.seed = seed
        self.dt = dt
        self.it = 0
        self.total_it = 0
        self.max_steps = MAX_STEPS
        self.bounds = 5.

        # Define actions and observation space
        self.action_space = gym.spaces.Box(
            low=-1., high=1., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-self.bounds, high=self.bounds, shape=(2,), dtype=np.float32)

        # init robot dart
        self.simu = rd.RobotDARTSimu(self.dt)
        self.simu.add_checkerboard_floor()
        self.simu.set_collision_detector("fcl")

        # configure robot
        self.robot = rd.Tiago()
        self.simu.add_robot(self.robot)

        # set robot initial pose
        self.reset_robot()

        # Control base - make the base fully controllable
        self.robot.set_actuator_type("servo", "rootJoint", False, True, False)

        # set target
        self.target = np.array([-1., 0.])

        # set init position
        self.init_obs = self.robot.base_pose().translation()[0:2]

        # add target visualization
        target_tf = dartpy.math.Isometry3()
        target_tf.set_translation([-1., 0., 0.])
        self.simu.add_visual_robot(rd.Robot.create_ellipsoid(
            [0.4, 0.4, 0.4], target_tf, 'fixed', 1.0, [0.0, 1.0, 0.0, 1.0], "target"))

        # add obstacle π shape
        obstacle_tf = dartpy.math.Isometry3()
        obstacle_tf.set_translation([0., 0., 0.25])
        obstacle_tf.set_rotation(
            dartpy.math.eulerXYZToMatrix([0, 0., np.pi/2.]))
        self.simu.add_robot(rd.Robot.create_box(
            [4.0, 1.0, 0.5], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "obstacle"))

        # imitate bouning box with spheres
        # temp_tf = dartpy.math.Isometry3()

        # temp_tf.set_translation([-2.5, 1.5, 0.5])
        # simu.add_visual_robot(rd.Robot.create_ellipsoid([0.2,0.2,0.2], temp_tf, 'fixed', 1000.0, [0., 1., 0., 1.0], "obstacle_1"))

        # temp_tf.set_translation([2.5, 0.5, 0.5])
        # simu.add_visual_robot(rd.Robot.create_ellipsoid([0.2,0.2,0.2], temp_tf, 'fixed', 1000.0, [0., 1., 0., 1.0], "obstacle_1"))
        # temp_tf.set_translation([2.5, -0.5, 0.5])
        # simu.add_visual_robot(rd.Robot.create_ellipsoid([0.2,0.2,0.2], temp_tf, 'fixed', 1000.0, [0., 1., 0., 1.0], "obstacle_1"))

        obstacle_tf.set_translation([-1., 2., 0.25])
        self.simu.add_robot(rd.Robot.create_box(
            [1.0, 3.0, 0.5], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "obstacle_2"))

        obstacle_tf.set_translation([-1., -2., 0.25])
        self.simu.add_robot(rd.Robot.create_box(
            [1.0, 3.0, 0.5], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "obstacle_3"))

        # # create wall
        # obstacle_tf.set_translation([-5., -0., 1.0])
        # self.simu.add_robot(rd.Robot.create_box(
        #     [10., 0.5, 2.0], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "wall_1"))

        # obstacle_tf.set_translation([5., -0., 1.0])
        # self.simu.add_robot(rd.Robot.create_box(
        #     [10., 0.5, 2.0], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "wall_2"))

        # obstacle_tf.set_translation([0., -5., 1.0])
        # self.simu.add_robot(rd.Robot.create_box(
        #     [0.5, 10.0, 2.0], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "wall_3"))

        # obstacle_tf.set_translation([0., 5., 1.0])
        # self.simu.add_robot(rd.Robot.create_box(
        #     [0.5, 10.0, 2.0], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "wall_4"))

        # self.history = History()
        # self.episode_history = np.array(self.state.copy())
        if (enable_graphics):
            self.render()

        self.episode_reward = 0.

    def step(self, action):
        self.it += 1
        self.state_prev = self.robot.base_pose().translation()[0:2]
        self.robot.set_commands(action, ['rootJoint_pos_x', 'rootJoint_pos_y'])

        self.simu.step_world()
        self.state = self.robot.base_pose().translation()[0:2]
        observation = self.state.copy()

        # # update episode history
        # self.episode_history = np.append(
        #     self.episode_history, [observation], axis=0)

        # calculate the distance to the target
        dist = np.linalg.norm(self.target - observation)

        # iters = int(self.total_it / (self.max_steps * 10))
        # max_iters = 100
        # p = (1. - min(1., iters/float(max_iters)))*0.6 + 0.4
        # print(p)
        # reward_dist = np.exp(-0.5 * dist * dist / (p * p))
        reward_dist = -dist * dist

        # calculate the exploration reward
        reward_exp = np.linalg.norm(self.init_obs - observation)#self.history._get_closest_point_dist(observation)
        if (self.init_obs - observation)[0] < 0.:
            reward_exp = 0.#-reward_exp * reward_exp
        else:
            reward_exp = reward_exp * reward_exp

        w0 = 4.
        w1 = 1. #1000. / (iters + 1)

        # calculate the reward (shift weights based on iteration number)
        reward = w0 * reward_dist + w1 * reward_exp

        done = False

        self.episode_reward += reward

        if (self.it == self.max_steps) or (abs(self.state[0]) >= self.bounds or abs(self.state[1]) >= self.bounds):
            done = True
            print(self.episode_reward)
            self.episode_reward = 0.
            # # update history every 10 episodes
            # if (self.total_it % (self.max_steps * 10) == 0):
            #     for point in self.episode_history:
            #         if (not self.history._in_hist(point)):
            #             self.history.add_point(point)
            #     # print(self.history._points)
            #     print("History updated")
            self.total_it += self.it
            self.it = -1

        return observation, reward, done, {}

    def reset(self):
        self.it = 0
        self.reset_robot()
        return self.state

    def reset_robot(self):
        self.robot.reset()

        arm_dofs = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint", "gripper_finger_joint", "gripper_right_finger_joint"]
        self.robot.set_positions(np.array([np.pi/2., np.pi/4., 0., np.pi/2., 0. , 0., np.pi/2., 0.03, 0.03]), arm_dofs)

        tf = self.robot.base_pose()
        self.robot.set_position_enforced(True)
        translation = tf.translation()
        translation[0] = 2.5
        tf.set_translation(translation)
        self.robot.set_base_pose(tf)
        self.state = self.robot.base_pose().translation()[0:2].reshape(1, 2)
        self.robot.set_positions([np.pi], ['rootJoint_rot_z'])
        self.robot.set_force_lower_limits(
            [-100., -100.], ['rootJoint_pos_x', 'rootJoint_pos_y'])
        self.robot.set_force_upper_limits(
            [100., 100, ], ['rootJoint_pos_x', 'rootJoint_pos_y'])
        # self.episode_history = np.array(self.state.copy())

    def render(self):
        if (self.enable_graphics):
            graphics = rd.gui.Graphics()
            self.simu.set_graphics(graphics)
            self.simu.scheduler().set_sync(False)
            graphics.look_at([-5, 3., 10.75], [0., 0., 0.])
            # graphics.record_video("Tiago.mp4")
        # print(self.state)


# class that holds the history of the robot's states, also has a grid to store the density of the states
# the grid is 100x100

class History:
    def __init__(self, max_capacity=4000):
        self._max_capacity = max_capacity

        self._points = []
        self._cache = {}

    def add_point(self, point):
        # point needs to be a np.array
        self._points.append(point)

        if len(self._points) > self._max_capacity:
            self._sparsify()

    def dist(self, point):
        d = None
        for i in range(len(self._points)):
            di = self._calc_dist(self._points[i], point)
            if d is None or di < d:
                d = di
        return d

    def _calc_dist(self, p1, p2):
        key = (p2[0], p2[1], p1[0], p1[1])
        if key in self._cache:
            return self._cache[key]
        key = (p1[0], p1[1], p2[0], p2[1])
        if key in self._cache:
            return self._cache[key]
        d = np.linalg.norm(p1 - p2)
        self._cache[key] = d

        return d

    def _sparsify(self):
        N = len(self._points)
        distances = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = self._calc_dist(self._points[i], self._points[j])
                distances[i, j] = d
                distances[j, i] = d

        while len(self._points) > self._max_capacity:
            i = self._get_most_dense_point(N, distances)
            distances = np.delete(distances, i, axis=0)
            distances = np.delete(distances, i, axis=1)
            del self._points[i]
            N = len(self._points)

    def _get_most_dense_point(self, N, distances, D=5):
        max_d = None
        max_i = None
        for i in range(N):
            dd = np.copy(distances[i, :])
            d = np.sum(np.sort(dd)[1:D+1])

            if max_i is None or d < max_d:
                max_d = d
                max_i = i
        return max_i

    def _in_hist(self, point):
        # find if point is in the history with a threshold of 0.1
        if (len(self._points) == 0):
            return False
        if (np.any(np.linalg.norm(self._points-point, axis=1) < 0.1)):
            return True
        else:
            return False

    def _get_closest_point_dist(self, point):
        # find the distance to the closest point in the history
        if (len(self._points) == 0):
            return 0
        else:
            a = np.min(np.linalg.norm(self._points-point, axis=1))
            # print(a)
            return a


env = TiagoEnv(enable_graphics=True)

model = algo(SACDensePolicy, env, verbose=1, learning_rate=5e-4, train_freq=int(MAX_STEPS/2), gradient_steps=200, batch_size=256, learning_starts=256)#, action_noise=NormalActionNoise(0., 1.))
# model = algo("MlpPolicy", env, verbose=1, learning_rate=5e-3)#, train_freq=MAX_STEPS, gradient_steps=200, batch_size=256, learning_starts=256)#, action_noise=NormalActionNoise(0., 1.))
# model = algo.load("tiago_lab_wall_new_reward")
model.learn(total_timesteps = 800 * 1000)
model.save("tiago_lab_new_reward")


obs = env.reset()
for i in range(env.max_steps):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action.reshape(2,))
    if i == 0:
        env.render()
    if done:
        break
