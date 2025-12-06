import numpy as np
from utils.state_vector_utils import polar_to_cartesian, cartesian_to_polar, calc_OE_from_cart


class Spacecraft:
    def __init__(self, r, theta, r_dot, v_theta, mass, C1, C2):
        # Initialize the state of the spacecraft
        self.update_state_polar(r, theta, r_dot, v_theta, mass)

        # Set propulsion parameters
        self.max_thrust = C1
        self.specific_impulse = C2

    def update_state_polar(self, r, theta, r_dot, v_theta, mass):
        # Set state vector polar coordinates coordinates
        self.r = r
        self.theta = theta
        self.r_dot = r_dot
        self.v_theta = v_theta

        # Set the spacecraft mass
        self.mass = mass

        # convert polar coordinates to cartesian
        x, y, vx, vy = polar_to_cartesian(r, theta, r_dot, v_theta)

        # Set state vector cartesian coordinates
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def update_state_cartesian(self, x, y, vx, vy, m):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = m

        r, theta, rdot, v_theta = cartesian_to_polar(x, y, vx, vy)

        # update polar coordinates
        self.r = r
        self.theta = theta
        self.r_dot = rdot
        self.v_theta = v_theta

    def calc_Planar_OE(self, x_cb, y_cb, vx_cb, vy_cb, mu_cb):

        # determine coordinates relative to central body
        x_rel = self.x - x_cb
        y_rel = self.y - y_cb
        vx_rel = self.vx - vx_cb
        vy_rel = self.vy - vy_cb

        a, e, w, theta = calc_OE_from_cart(
            x_rel, y_rel, vx_rel, vy_rel, mu_cb
        )

        return a, e, w, theta

    def get_cartesian_state(self):
        return self.x, self.y, self.vx, self.vy
