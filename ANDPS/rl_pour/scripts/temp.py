import RobotDART as rd
import numpy as np
from utils import damped_pseudoinverse
simu = rd.RobotDARTSimu(0.001)

def set_initial_pos(robot):
    robot.reset()
    positions = robot.positions()
    positions[4] = np.pi / 2.0
    positions[5] = np.pi / 2.0
    positions[6] = 3*np.pi/4
    positions[7] = 0.018
    positions[8] = 0.018
    robot.set_positions(positions)


def add_cereal(cereal_box, simu, count = 2):
    box_tf = robot.body_pose("cereal_box")
    cereal_arr = []
    box_tf = cereal_box.base_pose()
    cereal_dims = [0.01, 0.01, 0.01]
    cereal_mass = 0.1
    cereal_color = [1, 0., 0., 1.]


    for i in range(count):
        cereal_pose = [0., 0., 0., box_tf.translation()[0], box_tf.translation()[1]-0.09 , box_tf.translation()[2] + 0.2 + i * 0.01]
        cereal = rd.Robot.create_box(cereal_dims, cereal_pose, "free", mass=cereal_mass, color=cereal_color, box_name="cereal " + str(i))
        cereal_arr.append(cereal)
        simu.add_robot(cereal)

    return


# table
table_packages = [("table", "urdfs/table")]
table = rd.Robot("urdfs/table/table.urdf",  table_packages,"table")
table.set_color_mode("material")

table.fix_to_world()
# Graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 400
gconfig.height = 320

graphics = rd.gui.Graphics(gconfig)
simu.set_graphics(graphics)
graphics.look_at((1.5, 1.5, 4.), (0.0, 0.0, 0.5))

simu.add_checkerboard_floor()
simu.set_collision_detector('fcl')
simu.add_robot(table)

# robot
franka_packages = [("franka_description", "urdfs/franka/franka_description")]
robot = rd.Robot("urdfs/franka/franka.urdf",  franka_packages,"franka")
robot.set_color_mode("material")
# robot =
simu.add_robot(robot)
tf = robot.base_pose()
tf.set_translation([-0.7, 0, 0.78])
robot.set_base_pose(tf)
robot.fix_to_world()
robot.set_position_enforced(True)
robot.set_actuator_types("servo")
set_initial_pos(robot)
# cereal box
cereal_packages = [("cereal", "urdfs/cereal")]
cereal = rd.Robot("urdfs/cereal/cereal.urdf",  cereal_packages,"cereal")
cereal.set_color_mode("material")



tf = cereal.base_pose()
tf.set_translation(robot.body_pose("panda_ee").translation() + [0.0,0.1,-0.02])
cereal.set_base_pose(tf)
# simu.add_robot(cereal)
add_cereal(cereal,simu)




# bowl box
bowl_packages = [("bowl", "urdfs/bowl")]
bowl = rd.Robot("urdfs/bowl/bowl.urdf",  bowl_packages,"bowl")
bowl.set_color_mode("material")

for _ in range(1000):
    simu.step_world()
    # robot.set_commands([1*(0.015 - robot.positions(['panda_finger_joint1'])[0])],['panda_finger_joint1'])
tf = bowl.base_pose()
tf.set_translation(table.base_pose().translation() + [0, 0, 0.8])
bowl.set_base_pose(tf)
bowl.fix_to_world()
simu.add_robot(bowl)
eef_link_name = 'panda_ee'
for _ in range(2000000):

    robot.set_commands([-1],['panda_joint7'])
    robot.set_commands([-1],['panda_joint3'])
    simu.step_world()

