import numpy as np
class PIDTask:
    def __init__(self, dt, Kp=10., Ki=0.01, Kd=0.1):
        self._target = None
        self._dt = dt
        self._Kp = Kp
        self._Kd = Kd
        self._Ki = Ki
        self._sum_error = 0
        self._prev_error = None

    def set_target(self, target):
        self._target = target
        self._sum_error = 0
        self._prev_error = None

    def error(self, tf):
        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]

    def update(self, current):
        error_in_world_frame = self.error(current)
        if self._prev_error is None:
            derror = np.zeros_like(error_in_world_frame)
        else:
            derror = (error_in_world_frame - self._prev_error) / self._dt
        self._prev_error = error_in_world_frame

        self._sum_error = self._sum_error + error_in_world_frame * self._dt
        return self._Kp * error_in_world_frame + self._Ki * self._sum_error + self._Kd * derror, self.target_reached(error_in_world_frame)

    def target_reached(self, error):
        if np.linalg.norm(error[3:]) < 0.02:
            return True
        return False

    def set_gains(self, Kp, Ki, Kd):
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd


def damped_pseudoinverse(jac, l=0.1):
    m, n = jac.shape
    if n >= m:
        return jac.T @ np.linalg.inv(jac @ jac.T + l*l*np.eye(m))
    return np.linalg.inv(jac.T @ jac + l*l*np.eye(n)) @ jac.T