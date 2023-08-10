import numpy as np
import gym
import dartpy
import RobotDART as rd
import copy
import matplotlib.pyplot as plt
from utils import PIDTask as PID, damped_pseudoinverse


class PushEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_graphics=True, enable_record=True, seed=-1, dt=0.01, max_steps = 500):

        self.setup_simu(dt)
        if (enable_graphics):
            self.setup_graphics(enable_record)
        self.setup_env()

        self.seed = seed
        self.it = 1
        self.max_steps = max_steps

        # define action space
        self.action_space = gym.spaces.Box(low=np.array([-1., -1., -1.], dtype=np.float32), high=np.array([1., 1., 1.], dtype=np.float32), shape=(3,), dtype=np.float32)
        self.low_bounds = np.array([-2.,-2., 0.7], dtype=np.float32)
        self.high_bounds = np.array([2., 2., 3.], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.low_bounds, high=self.high_bounds, shape=(3,), dtype=np.float32)
        self.traj = []
        self.episode_rewards = []
        self.rewards_vel = []
        self.rewards_eef = []
        self.rewards_star = []
        self.max_reward = -np.inf
    def step(self, action):
        # action is the 3D cartesian velocity of the end effector, we keep the orientation fixed with a PID controller
        vel_rot = self.controller.update(self.robot.body_pose(self.eef_link_name))[0][:3]

        vel = np.concatenate((vel_rot, action))
        jac_pinv = damped_pseudoinverse(self.robot.jacobian(self.eef_link_name))
        cmd = jac_pinv @ vel
        self.robot.set_commands(cmd)
        self.simu.step_world()
        self.traj.append(copy.copy(self.box.base_pose().translation()))
        observation = copy.copy(self.get_state())
        reward = self.calc_reward()
        self.episode_rewards.append(reward)
        done = False
        # print("Step: ", str(self.it), "Out Of: ", str(self.max_steps), "Reward: ", str(reward))
        if (self.it == self.max_steps):
            done = True
            self.it = 0
            plt.plot(self.episode_rewards, label="total_reward")
            plt.plot(self.rewards_vel, label="vel_reward")
            plt.plot(self.rewards_eef, label="eef_reward")
            plt.plot(self.rewards_star, label="star_reward")
            plt.legend()
            total_reward = sum(self.episode_rewards)
            plt.title("Reward: " + str(total_reward))
            plt.savefig("plots/cur_reward.png")
            if(total_reward > self.max_reward):
                self.max_reward = total_reward
                plt.savefig("plots/max_reward.png")
            plt.close()
            plt.cla()
            plt.clf()
            self.traj = []
            self.episode_rewards = []
            self.rewards_eef = []
            self.rewards_vel = []
            self.rewards_star = []

        self.it += 1

        return observation, reward, done, {}

    def get_state(self):
        return (self.robot.body_pose(self.eef_link_name).translation()).astype(np.float32)

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
        self.graphics.look_at([0., 3., 2.75], [0., 0., 0.75])
        if (enable_record):
            self.graphics.camera().record(True)
            self.graphics.record_video("push.mp4", self.simu.graphics_freq())

    def setup_table(self):
        table_packages = [("table", "robots/table")]
        self.table = rd.Robot("robots/table/table.urdf",
                              table_packages, "table")
        self.table.set_color_mode("material")
        self.table.fix_to_world()
        self.simu.add_robot(self.table)

    def setup_robot(self):
        # Iiwa custom eef
        iiwa_packages = [("iiwa_description", "robots/iiwa/iiwa_description")]
        self.robot = rd.Robot("robots/iiwa/iiwa.urdf", iiwa_packages, "iiwa")
        robot_base_pose = [0., 0., np.pi/2., 0., -0.8, 0.75]
        self.robot.set_base_pose(robot_base_pose)

        self.simu.add_robot(self.robot)
        self.eef_link_name = "iiwa_link_ee"
        self.robot.fix_to_world()
        self.robot.set_position_enforced(True)
        self.robot.set_actuator_types("servo")
        self.reset_robot()

    def reset_robot(self):
        self.robot.reset()
        init_positions = copy.copy(self.robot.positions())
        # init_positions[0] = -2.
        init_positions[3] = -np.pi / 2.0
        init_positions[5] = np.pi / 4.0
        init_positions[1] = np.pi / 4.0
        self.robot.set_positions(init_positions)
        Kp = np.array([20., 20., 20., 10., 10., 10.])
        Kd = Kp * 0.01
        Ki = Kp * 0.1
        self.controller = PID(self.dt, Kp, Ki, Kd)
        self.controller.set_target(self.robot.body_pose(self.eef_link_name))

    def setup_star(self):
        box_packages = [("star", "robots/star")]
        self.box = rd.Robot("robots/star/star.urdf",   box_packages, "star")
        self.box.set_base_pose([0., 0., 0.5,  0., -0.1, 0.78])
        self.simu.add_robot(self.box)

    def reset_star(self):
        self.box.reset()
        self.box.set_base_pose([0., 0., 0.5,  0., -0.1, 0.78])

    def setup_target(self):
        self.target = rd.Robot.create_ellipsoid([0.25, 0.25, 0.025], [
            0., 0., 0., 0.5, 0.5, 0.8], "fixed", mass=0.01, color=[0.0, 1.0, 0.0, 0.5], ellipsoid_name="target")
        self.simu.add_visual_robot(self.target)

    def reset_target(self):
        target_tf = dartpy.math.Isometry3()
        target_tf.set_translation([-0.1, 0.12, 0.78])
        self.target.set_base_pose(target_tf)

    def setup_env(self):
        print("Initializing Rd Environment")
        self.setup_table()
        print("Added table")
        self.setup_robot()
        print("Added robot")
        self.setup_star()
        print("Added star")
        self.setup_target()
        print("Added target")

    def reset(self):
        self.reset_robot()
        self.reset_star()
        self.reset_target()
        print("Env reset successfully")
        return self.get_state()

    def calc_reward(self):

        star_to_center = -np.linalg.norm(self.box.base_pose().translation() - self.target.base_pose().translation())
        # p = 0.1
        # star_to_center = np.exp(-0.5 * dist * dist / (p * p))

        eef_to_star = -0.1 * np.linalg.norm(self.robot.body_pose(self.eef_link_name).translation() - self.box.base_pose().translation())
        vel = -0.01 * np.linalg.norm(self.robot.body_velocity(self.eef_link_name)[3:])

        self.rewards_eef.append(eef_to_star)
        self.rewards_vel.append(vel)
        self.rewards_star.append(star_to_center)

        reward = star_to_center + eef_to_star + vel

        return reward
