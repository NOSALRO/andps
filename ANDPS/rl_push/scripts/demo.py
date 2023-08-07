import RobotDART as rd
import copy
import numpy as np
# Simu
simu = rd.RobotDARTSimu(0.01)
simu.set_collision_detector("fcl")



# Graphics
graphics = rd.gui.Graphics()
simu.set_graphics(graphics)
# graphics.look_at([0.5, 3., 0.75], [0.5, 0., 0.2])

# Iiwa custom eef
iiwa_packages = [("iiwa_description", "robots/iiwa/iiwa_description")]
robot = rd.Robot("robots/iiwa/iiwa.urdf", iiwa_packages, "iiwa")

init_positions = copy.copy(robot.positions())
# init_positions[0] = -2.
init_positions[3] = -np.pi / 2.0
init_positions[5] = np.pi / 2.0
robot.set_positions(init_positions)
robot.fix_to_world()
robot.set_position_enforced(True)
robot.set_actuator_types("servo")





#  Table







simu.add_robot(robot)
simu.add_checkerboard_floor()
simu.run(50.)
