import numpy as np
import gym
import dartpy
import RobotDART as rd
from utils import angle_wrap


class TiagoEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_graphics=False, seed=-1, dt=0.01, max_steps=500):
        super(TiagoEnv, self).__init__()
        self.enable_graphics = enable_graphics

        # x, y, theta
        self.state = np.array([[0., 0., 0.]])

        self.seed = seed
        self.dt = dt
        self.it = 0
        self.total_episodes = 0
        self.max_steps = max_steps
        self.bounds = 5.

        # Define actions and observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.observation_space = gym.spaces.Box(low=np.array([-self.bounds, -self.bounds, -np.pi]), high=np.array([self.bounds,self.bounds,np.pi]), shape=(3,))

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
        self.target = np.array([-1, -1.2, -np.pi/2])

        # set init position
        self.init_obs = self.robot.base_pose().translation()[0:2]

        # add target visualization
        target_tf = dartpy.math.Isometry3()
        target_tf.set_translation([-1., -1.2, 0.])
        self.simu.add_visual_robot(rd.Robot.create_ellipsoid([0.4, 0.4, 0.4], target_tf, 'fixed', 1.0, [0.0, 1.0, 0.0, 1.0], "target"))

        # add counter
        counter_tf = dartpy.math.Isometry3()
        counter_tf.set_translation([-1., -2., 0.25])
        # counter_tf.set_rotation(dartpy.math.eulerXYZToMatrix([0, 0., np.pi/2.]))
        self.simu.add_robot(rd.Robot.create_box([4.0, 1.0, 0.5], counter_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "counter"))

        if (enable_graphics):
            self.render()

        self.episode_reward = 0.

    def step(self, action):
        self.it += 1
        self.state_prev = self.robot.base_pose().translation()[0:2]
        self.robot.set_commands(action[:-1], ['rootJoint_pos_x', 'rootJoint_pos_y'])
        self.robot.set_commands([action[-1]], ['rootJoint_rot_z'])
        self.simu.step_world()
        self.state = self.get_state()
        observation = self.state.copy()

        # # update episode history
        # self.episode_history = np.append(
        #     self.episode_history, [observation], axis=0)

        # calculate the distance to the target
        dist = np.linalg.norm(self.target[0:2] - observation[0:2])

        # iters = int(self.total_it / (self.max_steps * 10))
        # max_iters = 100
        # p = (1. - min(1., iters/float(max_iters)))*0.6 + 0.4
        # # print(p)
        # reward_pos = np.exp(-0.5 * dist * dist / (p * p))
        reward_pos = -dist * dist

        # calculate the exploration reward
        # reward_exp = np.linalg.norm(self.init_obs - observation)#self.history._get_closest_point_dist(observation)
        # if (self.init_obs - observation)[0] < 0.:
        #     reward_exp = 0.#-reward_exp * reward_exp
        # else:
        #     reward_exp = reward_exp * reward_exp
        reward_rot = -np.sin(observation[2])
        w0 = 10.
        w1 = 1.
        # if(self.it> self.max_steps * 0.7):
        #     w0 *= 10
        #     w1*=2

        # calculate the reward (shift weights based on iteration number)
        reward = w0 * reward_pos + w1 * reward_rot

        done = False

        self.episode_reward += reward

        if (self.it == self.max_steps) or (abs(self.state[0]) >= self.bounds or abs(self.state[1]) >= self.bounds):
            done = True
            print("Reward: ", reward, "Pos: ", reward_pos, "Rot: ", reward_rot)
            print("Final State: ", self.state)
            print("Target State:",self.target)
            self.episode_reward = 0.
            # # update history every 10 episodes
            # if (self.total_it % (self.max_steps * 10) == 0):
            #     for point in self.episode_history:
            #         if (not self.history._in_hist(point)):
            #             self.history.add_point(point)
            #     # print(self.history._points)
            #     print("History updated")
            self.total_episodes += 1
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
        translation[0] = 2.

        tf.set_translation(translation)
        self.robot.set_base_pose(tf)
        self.state = self.get_state()

        self.robot.set_force_lower_limits(
            [-100., -100.], ['rootJoint_pos_x', 'rootJoint_pos_y'])
        self.robot.set_force_upper_limits(
            [100., 100, ], ['rootJoint_pos_x', 'rootJoint_pos_y'])
        # self.robot.set_positions([0], ['rootJoint_rot_z'])
        # self.episode_history = np.array(self.state.copy())
    def get_state(self):
        state = np.zeros( 3)
        state[0:2] = self.robot.base_pose().translation()[0:2]
        state[2] = angle_wrap(self.robot.positions(['rootJoint_rot_z'])[0])
        return state

    def render(self):
        if (self.enable_graphics):
            graphics = rd.gui.Graphics()
            self.simu.set_graphics(graphics)
            self.simu.scheduler().set_sync(False)
            graphics.look_at([-5, 3., 10.75], [0., 0., 0.])
            graphics.record_video("Tiago.mp4")
        # print(self.state)
