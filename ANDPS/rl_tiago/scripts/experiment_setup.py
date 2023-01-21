import RobotDART as rd
import dartpy
import numpy as np



def reset_robot():
    robot.reset()
    tf = robot.base_pose()
    robot.set_position_enforced(True)
    translation = tf.translation()
    translation[0] = 2.5

    tf.set_translation(translation)
    robot.set_base_pose(tf)
    state = robot.base_pose().translation()[0:2].reshape(1,2)
    robot.set_positions([0], ['rootJoint_rot_z'])

dt = 0.001
# init robot dart
simu = rd.RobotDARTSimu(dt)
simu.add_checkerboard_floor()
simu.set_collision_detector("fcl")
graphics = rd.gui.Graphics()
simu.set_graphics(graphics)
graphics.look_at([0.5, 5., 0.75], [0.5, 0., 0.2])
graphics.record_video("Talos.mp4")

# configure robot
robot = rd.Tiago()
simu.add_robot(robot)

# set robot initial pose
reset_robot()

# Control base - make the base fully controllable
robot.set_actuator_type("servo", "rootJoint", False, True, False)

# set target
target = np.array([-1., 0.])

# add target visualization
target_tf = dartpy.math.Isometry3()
target_tf.set_translation([-1., 0., 0.])
simu.add_visual_robot(rd.Robot.create_ellipsoid([0.4, 0.4, 0.4], target_tf, 'fixed', 1.0, [0.0, 1.0, 0.0, 1.0], "target"))





# add obstacle π shape
obstacle_tf = dartpy.math.Isometry3()
obstacle_tf.set_translation([0., 0., 0.])
obstacle_tf.set_rotation(dartpy.math.eulerXYZToMatrix( [0, 0.,np.pi/2.]))


simu.add_robot(rd.Robot.create_box([4.0, 1.0, 3.0], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "obstacle"))
obstacle_tf.set_translation([-1., 2., 0.])
simu.add_robot(rd.Robot.create_box([1.0, 3.0, 3.0], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "obstacle_2"))
obstacle_tf.set_translation([-1., -2., 0.])
simu.add_robot(rd.Robot.create_box([1.0, 3.0, 3.0], obstacle_tf, 'fixed', 1000.0, [0.8, 0.8, 0.8, 1.0], "obstacle_3"))


# add bounding box for tiago
# robot.create_box([0.5, 0.5, 2.], robot.base_pose(), 'fcl', 1.0, [0.8, 0.8, 0.8, 1.0], "bounding_box")
# simu.add_robot(tiago_mbr)


# set collison mask
for _ in range (8000):

    A = np.array([[1, -1], [1, 1]])

    action = A@(target - robot.positions(['rootJoint_pos_x', 'rootJoint_pos_y']))
    robot.set_commands(action, ['rootJoint_pos_x', 'rootJoint_pos_y'])
    simu.step_world()