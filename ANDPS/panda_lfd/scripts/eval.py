import RobotDART as rd
import numpy as np
import dartpy
from utils import net_rd as andps, simple_nn, torch, damped_pseudoinverse, PIDTask
# from generate_datasets import get_joints_state, get_eef_state, set_initial_pos
# Get Joint Positions (Angles)
def get_joints_state(robot):
    return robot.positions()[:7]

# get end effector position (xyz)
def get_eef_state(robot):
    return robot.body_pose(eef_link_name).translation()

# Set Franka to Initial Configuration
def set_initial_pos(robot):
    robot.reset()
    positions = robot.positions()
    positions[5] = np.pi / 2.0
    positions[6] = np.pi / 4.0
    positions[7] = 0.04
    positions[8] = 0.04
    robot.set_positions(positions)


enable_graphics = True # set to False to run without graphics

experiment = "eef"
# experiment = "joint"

policy = "ANDPS"
# policy = "NN"

# read dataset
dataset = np.load("data/spiral_" + experiment + "_space.npy")
if experiment == "eef":
    np_X = dataset[:, :3]
    np_Y = dataset[:, -3:]
elif experiment == "joint":
    np_X = dataset[:, :7]
    np_Y = dataset[:, -7:]

target = np_X[-1, :]
dims = np_X.shape[1]
if policy == "ANDPS":
    model = andps(dims, 6, target)

elif policy == "NN":
    model = simple_nn(dims)

model.load_state_dict(torch.load("models/"+policy+"_spiral_" + experiment + "_1000.pt"))
model.eval()


# RobotDART Simulation
dt = 0.001
control_freq = 100
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("fcl")
simu.add_checkerboard_floor()

# RobotDART Graphics
if(enable_graphics):
    gconfig = rd.gui.Graphics.default_configuration()
    gconfig.width = 640
    gconfig.height = 480
    graphics = rd.gui.Graphics(gconfig)
    simu.set_graphics(graphics)
    graphics.look_at((2.5, 0.0, 1.), (0.0, 0.0, 0.5))
    graphics.camera().record(True)
    simu.enable_status_bar(enable=False)
    graphics.record_video("media/videos/spiral_eval_"+experiment+".mp4")



# RobotDART Robot
robot = rd.Franka()
robot.fix_to_world()
robot.set_position_enforced(True)
robot.set_actuator_types("servo")
eef_link_name = "panda_ee"
simu.add_robot(robot)

set_initial_pos(robot)
if experiment == "joint":
    # Initialize datasets
    joints_state_cur = get_joints_state(robot)
    joints_dataset = np.array(joints_state_cur)
    get_state = get_joints_state
elif experiment == "eef":
    eef_state_cur = get_eef_state(robot)
    eef_trajectory = np.array(eef_state_cur)
    get_state = get_eef_state


# Controller for fixed eef orientation
Kp = np.array([20., 20., 20., 10., 10., 10.])
Kd = Kp * 0.01
Ki = Kp * 0.1
controller = PIDTask(1./control_freq, Kp, Ki, Kd)
controller.set_target(robot.body_pose(eef_link_name))

#  visualize target
tf_target = dartpy.math.Isometry3()
tf_target.set_translation(target)
simu.add_visual_robot(rd.Robot.create_ellipsoid([0.05, 0.05, 0.05], tf_target, "fix",  color=[0, 1, 0., 1], ellipsoid_name='target'))

# helper variables
timestep_counter = 0
t = 0
leave_trace = 0
continue_simu = True
x_cur = get_eef_state(robot)
with torch.no_grad():
    while(np.linalg.norm(get_state(robot)-target)>1e-2):
        t += dt
        update = False
        if simu.scheduler().schedule(control_freq):
            update = True
            if(experiment == "eef"):
                vel_rot = controller.update(robot.body_pose(eef_link_name))[0][:3]
                vel_xyz = model(torch.Tensor(x_cur).view(-1,dims)).detach().numpy().copy()
                vel = [vel_rot[0], vel_rot[1], vel_rot[2], vel_xyz[0,0], vel_xyz[0,1], vel_xyz[0,2]]
                # Jacobian pseudo-inverse
                jac_pinv = damped_pseudoinverse(robot.jacobian(eef_link_name))
                # Compute joint commands
                cmd = jac_pinv @ vel
            elif(experiment == "joint"):
                joints_vel = model(torch.Tensor(get_joints_state(robot)).view(-1,dims)).detach().numpy()
                cmd = np.append(joints_vel, np.zeros(2))


            robot.set_commands(cmd)

            # Add spheres to visualize the trajectory in the GUI
            if(enable_graphics):
                leave_trace = leave_trace + 1
                if leave_trace == 1 or leave_trace % 10 == 0:
                    simu.add_visual_robot(rd.Robot.create_ellipsoid([0.05, 0.05, 0.05], robot.body_pose(
                    eef_link_name), "fix",  color=[1, 0.647, 0., 1], ellipsoid_name=str(leave_trace)))

        if(update):
            x_cur = get_state(robot)

        simu.step_world()
        # if update:
        #     joints_state_cur = get_joints_state(robot)
        #     joints_dataset = np.vstack((joints_dataset, joints_state_cur))
        #     eef_trajectory = np.vstack((eef_trajectory, robot.body_pose(eef_link_name).translation()))
