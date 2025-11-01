import numpy as np
from utils.state_vector_utils import polar_to_cartesian, cartesian_to_polar


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

        x_rel_km = x_rel / 1000
        y_rel_km = y_rel / 1000

        # position and velocity magnitudes
        r_km = (x_rel_km**2 + y_rel_km**2) ** 0.5
        r = r_km * 1000  # convert back to m


        # spacecraft position, vel, and z vectors
        sc_pos = np.array([x_rel, y_rel, 0.0])
        sc_vel = np.array([vx_rel, vy_rel, 0.0])
        z_hat = np.array([1.0, 0.0, 0.0])
        r_hat = sc_pos / r

        # angular momentum
        h_vec = np.cross(sc_pos, sc_vel)
        h = np.linalg.norm(h_vec)
        h_hat = h_vec / h

        # node line
        N = np.cross(z_hat, h_hat)
        N_hat = N / np.linalg.norm(N)

        # eccentricity vector
        e_vec = np.cross(sc_vel, h_vec) / mu_cb - sc_pos / r
        e = np.linalg.norm(e_vec)

        # If e is zero, we will get an error dividing by zero, so the ecc vector
        # is set at {0,0,0} if the magnitude is zero.
        if e == 0.0:
            e_hat = e_vec * 0.0
        else:
            e_hat = e_vec / e

        # semi major axis
        rp = h**2 / mu_cb / (1 + e * np.cos(0))
        ra = h**2 / mu_cb / (1 + e * np.cos(np.pi))
        a = 1 / 2 * (rp + ra)

        # argument of periapsis
        if e_vec[2] >= 0.0:
            w = np.acos(np.dot(N_hat, e_hat))
        else:
            w = 2 * np.pi - np.acos(np.dot(N_hat, e_hat))

        # true anomaly - extra error handling included,
        # mainly needed for hyperbolic instances
        if np.dot(sc_pos, sc_vel) >= 0.0:
            dotp = np.dot(e_hat, r_hat)
            if dotp < -1:
                dotp = -1
            elif dotp > 1:
                dotp = 1

            theta = np.acos(dotp)

        else:
            # check acos domain
            dotp = np.dot(e_hat, r_hat)
            if dotp < -1:
                dotp = -1
            elif dotp > 1:
                dotp = 1

            theta = 2 * np.pi - np.acos(dotp)

        return a, e, w, theta

    def get_cartesian_state(self):

        return self.x, self.y, self.vx, self.vy
