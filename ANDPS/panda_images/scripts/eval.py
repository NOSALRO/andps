import matplotlib.pyplot as plt
import RobotDART as rd
import numpy as np
import dartpy
from utils import *
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import andps_images as andps, CustomDataset, simple_cnn
from torchvision import transforms
convert_tensor = transforms.ToTensor()

# EXPERIMENT = 'ANDPS'
# EXPERIMENT_NICE_NAME = 'andps_images'
EXPERIMENT = 'CNN'
EXPERIMENT_NICE_NAME = 'simple_cnn'

TARGET_MOTION = "angle"
# TARGET_MOTION = "sine"
# TARGET_MOTION = "line"

# Test ANDPs' Robustness
APPLY_PERTURBATION = True
PERTURBATION_TYPE = "push"
# PERTURBATION_TYPE = "noise"
NOISE_PERCENTAGE = 1
# NOISE_PERCENTAGE = 3


# Test ANDPs' reactiveness
CHANGE_IMAGE = False
TARGET_MOTION_TWO = "sine"
if CHANGE_IMAGE:
    assert TARGET_MOTION != TARGET_MOTION_TWO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# get end effector position (xyz)
def get_eef_state(robot, eef_link_name="panda_ee"):
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

# convert rd image to grayscale array


def image_as_grayscale_array(rd_image):
    image = np.array(rd.gui.convert_rgb_to_grayscale(
        rd_image).data).reshape(rd_image.width, rd_image.height)
    return image

def get_state(robot, rd_image):
    pos = robot.body_pose(eef_link_name).translation()
    if (PERTURBATION_TYPE == "noise" and APPLY_PERTURBATION):
        pos += np.random.normal(0, np.abs(pos) * NOISE_PERCENTAGE/100)
    img = np.array(rd.gui.convert_rgb_to_grayscale(rd_image).data)
    return np.concatenate([pos, img])

target = np.array([0.55, 0.404,  0.52])

# RobotDART Simulation
dt = 0.01
control_freq = 100
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("fcl")
simu.add_checkerboard_floor()

# RobotDART Robot
robot = rd.Franka()
robot.fix_to_world()
robot.set_position_enforced(True)
robot.set_actuator_types("servo")
eef_link_name = "panda_ee"
simu.add_robot(robot)
set_initial_pos(robot)


if (APPLY_PERTURBATION or CHANGE_IMAGE):
    assert CHANGE_IMAGE != APPLY_PERTURBATION

# box
box_packages = [(TARGET_MOTION+"_box", "urdfs/"+TARGET_MOTION+"_box")]
box = rd.Robot("urdfs/"+TARGET_MOTION+"_box/"+TARGET_MOTION +
                    "_box.urdf",  box_packages, TARGET_MOTION+"_box")
box.set_color_mode("material")
box.fix_to_world()
box_tf = box.base_pose()
box_tf.set_translation([3.5, 0., 1.])
box.set_base_pose(box_tf)
simu.add_robot(box)

# graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 480
gconfig.height = 480
gconfig.shadowed = False
graphics = rd.gui.Graphics(gconfig)
simu.set_graphics(graphics)
simu.enable_status_bar(enable=False)
graphics.look_at((-2.5, 1.0, 1.), (0.0, 0.0, 0.5))

# record images from main camera/graphics
graphics.camera().record(True)
graphics.record_video(EXPERIMENT_NICE_NAME +"_"+TARGET_MOTION+"-env-eval.mp4", simu.graphics_freq())

# add camera
im_width = 64
im_height = 64
camera = rd.sensor.Camera(graphics.magnum_app(), im_width, im_height)
camera.camera().record(True)
camera.record_video(EXPERIMENT_NICE_NAME +"_"+TARGET_MOTION+"-pov-eval.mp4")

# make camera look forward, attach to ee, add to simu
tf = dartpy.math.Isometry3()
rot = dartpy.math.AngleAxis(-np.pi/2, [0., 1., 0.])
rot = rot.multiply(dartpy.math.AngleAxis(
    1.57, [0., 0., 1.])).to_rotation_matrix()
tf.set_translation([0., 0., 0.1])
tf.set_rotation(rot)
camera.attach_to_body(robot.body_node("panda_ee"), tf)
simu.add_sensor(camera)

# Controller for fixed eef orientation
Kp = np.array([20., 20., 20., 10., 10., 10.])
Kd = Kp * 0.01
Ki = Kp * 0.1
controller = PIDTask(1./control_freq, Kp, Ki, Kd)
controller.set_target(robot.body_pose(eef_link_name))

# visualize target
target_tf = dartpy.math.Isometry3()
target_tf.set_translation(target)
simu.add_visual_robot(rd.Robot.create_ellipsoid([0.05, 0.05, 0.05], target_tf, "fix",  color=[
                      0.211, 0.596, 0.184, 1], ellipsoid_name="target"))

# warmstart simulation
for _ in range(10):
    simu.step_world()

# Initialize datasets
eef_trajectory = np.array(get_eef_state(robot))
images = np.array(image_as_grayscale_array(camera.image()))

# helper variables
timestep_counter = 0
t = 0
leave_trace = 0
continue_simu = True


# Number of dynamical systems
if EXPERIMENT == 'ANDPS':
    num_DS = 4
    net = andps(3, num_DS, target, device)
    net.load_state_dict(torch.load("models/panda_image_100.pt", map_location=device))
elif EXPERIMENT == 'CNN':
    net = simple_cnn(ds_dim=3)
    net.load_state_dict(torch.load("models/simple_cnn_100.pt", map_location=device))

net.to(device)

net.eval()
while (continue_simu):
    t += dt

    update = False
    if simu.scheduler().schedule(control_freq):
        # Apply perturbations
        if (APPLY_PERTURBATION):
            # noise perturbation is handled in the get_state function so we only deal with push here
            if (PERTURBATION_TYPE == "push"):
                if (0 <= t and t <= 0.2):
                    robot.set_external_force(eef_link_name, [-50., -100., -80.])
                elif (4 <= t and t <= 4.3):
                    robot.set_external_force(eef_link_name, [0., 100., 100.])
                else:
                    robot.set_external_force(eef_link_name, [0., 0., 0.])
            
        elif (CHANGE_IMAGE and 2.49<=t<=2.5):
            print("changing image")
            # let's remove the box
            simu.remove_robot(box)

            # And add the box with the new motion
            box_packages = [(TARGET_MOTION_TWO+"_box", "urdfs/"+TARGET_MOTION_TWO+"_box")]
            box = rd.Robot("urdfs/"+TARGET_MOTION_TWO+"_box/"+TARGET_MOTION_TWO +
                                "_box.urdf",  box_packages, TARGET_MOTION_TWO+"_box")
            box.set_color_mode("material")
            box.fix_to_world()
            box_tf = box.base_pose()
            box_tf.set_translation([3.5, 0., 1.])
            box.set_base_pose(box_tf)
            simu.add_robot(box)

        update = True
        eef_tf = robot.body_pose(eef_link_name)
        vel_rot = controller.update(eef_tf)[0][:3]

        vel_xyz = net(torch.Tensor(get_state(robot, camera.image(
        )).reshape(-1, 64*64 + 3)).to(device)).detach().cpu().numpy().copy()
        vel = np.concatenate((vel_rot, vel_xyz.reshape(3,)))

        # Jacobian pseudo-inverse
        jac_pinv = damped_pseudoinverse(robot.jacobian(eef_link_name))

        # Compute joint commands
        cmd = jac_pinv @ vel
        robot.set_commands(cmd)

        timestep_counter += 1
        if (timestep_counter == 1500):
            continue_simu = False
    simu.step_world()
    if update:
        eef_trajectory = np.vstack(
            (eef_trajectory, robot.body_pose(eef_link_name).translation()))
        images = np.dstack((images, image_as_grayscale_array(camera.image())))


data = np.load('data/'+TARGET_MOTION+'.npz')


if (APPLY_PERTURBATION and PERTURBATION_TYPE == "push"):
    np.savez('data/'+EXPERIMENT_NICE_NAME+"_"+TARGET_MOTION+'_push_eval.npz',
             train=data['np_X'][:, :3], test=eef_trajectory, images=images, t_application=[0, 4])
elif (APPLY_PERTURBATION and PERTURBATION_TYPE == "noise"):
    np.savez('data/'+EXPERIMENT_NICE_NAME+"_"+TARGET_MOTION+'_noise_'+str(NOISE_PERCENTAGE)+'_perc_eval.npz',
             train=data['np_X'][:, :3], test=eef_trajectory, images=images, noise_percentage=NOISE_PERCENTAGE)
elif (CHANGE_IMAGE):
    np.savez('data/'+EXPERIMENT_NICE_NAME+"_"+TARGET_MOTION+'_to_'+TARGET_MOTION_TWO+'_eval.npz',
             train=data['np_X'][:, :3], test=eef_trajectory, images=images, t_change=2.5)
else:
    np.savez('data/'+EXPERIMENT_NICE_NAME+"_"+TARGET_MOTION+'_eval.npz',
             train=data['np_X'][:, :3], test=eef_trajectory, images=images)
