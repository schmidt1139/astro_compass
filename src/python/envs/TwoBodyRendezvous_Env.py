import numpy as np
import gymnasium as gym
from typing import Optional
from scipy.integrate import solve_ivp
from constants.constants import Constants
from core.spacecraft import Spacecraft
from core.propagation import env_EOM_TBT_v2
from utils.state_vector_utils import polar_to_cartesian, calc_cart_from_OE, cartesian_to_polar


class TwoBodyRendezvous_Env(gym.Env):
    def __init__(self, **kwargs):
        # define limits of the state parameters
        low_array = np.full(
            9, -np.inf, dtype=np.float32
        )  # lower bounds for state space
        high_array = np.full(
            9, np.inf, dtype=np.float32
        )  # upper-bounds for state space

        # define the state space (in this case the observation is the state) 5x5
        self.observation_space = gym.spaces.Box(low=low_array, high=high_array)

        self._state = np.full(10, 0.0, dtype=np.float32)  # initialize state vector

        self._keplerian_elements = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

        # list of environment parameters (Sun is the central body)
        self.param_mu = kwargs.get("mu", Constants.MU_SUN_M)  # in m^3/s^2
        self.C1 = (
            kwargs.get("max_T", 1.33)
        )  # Spacecraft max thrust (in N)
        self.C2 = kwargs.get("ISP", 3872.0)  # Spacecraft specific impulse (s)
        self.l_star = kwargs.get(
            "l_star", Constants.SMA_EARTH
        )  # characteristic length (m)
        self.t_star = kwargs.get(
            "t_star", (Constants.SMA_EARTH**3 / (Constants.MU_SUN_M)**0.5)
        )  # characteristic time (s)
        self.m_star = kwargs.get("m_star", 3366.0)  # characteristic mass (kg)
        self.step_size = kwargs.get("step_size", 86400)  # environment step size (s)
        self.arr_mu = np.array([self.param_mu])  # solar mu [m^3/s^2]
        self.planet_radii = np.array([Constants.RADIUS_SUN_M])  # solar radius [m]
        self.elapsed_t = 0.0
        self.episode_reward = 0.0

        # define the action space
        # The action space consists of three variables:
        #    1) a control throlle input (scaled from 0 to 1)
        #    2) a thrust direction x vector component (ranges -1 to 1)
        #    3) a thrust direction y vector component (ranges -1 to 1)
        low_array_action = np.array([0.0, -1.0, -1.0], dtype=np.float32)
        high_array_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low_array_action, high=high_array_action)

    def _get_info(self, ode_solution, delta_r):
        # to-do: add orbital elements as optional and append to output dictionary

        #check periods
        a = self._keplerian_elements[0]
        mu = self.param_mu
        T = 2 * np.pi * ( a**3 / mu ) ** 0.5

        #calc deltas
        deltas = self.calc_deltas()
        dx, dy, dr, dvx, dvy, dv, r_target, v_target, dr_norm, dv_norm = deltas

        # target orbital elements
        x_target = self._state[6]
        y_target = self._state[7]
        vx_target = self._state[8]
        vy_target = self._state[9]
        r_target, theta_target, r_dot_target, v_theta_target = cartesian_to_polar(x_target, y_target, vx_target, vy_target)

        # Initialize a spacecraft object with the state of the environment
        sc_target = Spacecraft(r_target, theta_target, r_dot_target, v_theta_target, 1000.0, self.C1, self.C2)

        # calculate orbital elements
        a_target, e_target, w_target, theta_target = sc_target.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, mu)
        T_target = 2 * np.pi * (a_target**3 / mu) ** 0.5

        return {
            "Elapsed time": self.elapsed_t,
            "ODE Solution": ode_solution,
            "delta_state": delta_r,
            "planet_radii": self.planet_radii,
            "a": self._keplerian_elements[0]/Constants.SMA_EARTH,
            "e": self._keplerian_elements[1],
            "w": np.rad2deg(self._keplerian_elements[2]),
            "theta": np.rad2deg(self._keplerian_elements[3]),
            "max_thrust": self._spacecraft.max_thrust,
            "ISP": self._spacecraft.specific_impulse,
            "dx": dx,
            "dy": dy,
            "dr": dr,
            "dvx": dvx,
            "dvy": dvy,
            "dv": dv,
            "r_target": r_target,
            "v_target": v_target,
            "dr_norm": dr_norm,
            "dv_norm": dv_norm,
            "orbital_period_yrs": T/(365.25*24*60*60),
            "a_target": a_target/Constants.SMA_EARTH,
            "e_target": e_target,
            "w_target": np.rad2deg(w_target),
            "theta_target": np.rad2deg(theta_target),
            "orbital_period_target_yrs": T_target/(365.25*24*60*60)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # reset cumulative reward
        self.episode_reward = 0.0

        # extract central body gravitational parameter
        mu = self.param_mu

        # set ranges for initial state parameters
        a_range = [Constants.SMA_VENUS, Constants.SMA_MARS]  # initial radius range (m)
        e_range = [0.0, 0.75]  # initial eccentricity range
        w_range = [0.0, 2 * np.pi]  # initial argument of periapsis range (rad)
        theta_range = [0.0, 2 * np.pi]  # initial true anomaly range (rad)

        # set the initial spacecraft parameters
        mass = 3366.0  # Assumed spacecraft total mass
        C1 = 1.33  # Spacecraft max thrust (in kN)
        C2 = 3872.0  # Spacecraft specific impulse (s)

        # randomly vary initial state
        a_0 = self.np_random.uniform(low=a_range[0], high=a_range[1])
        e_0 = self.np_random.uniform(low=e_range[0], high=e_range[1])
        w_0 = self.np_random.uniform(low=w_range[0], high=w_range[1])
        theta_0 = self.np_random.uniform(low=theta_range[0], high=theta_range[1])

        # convert initial state to cartesian coordinates
        x_0, y_0, vx_0, vy_0 = calc_cart_from_OE(a_0, e_0, w_0, theta_0, mu)

        # convert initial state to polar coordinates
        r_0, theta_0, r_dot_0, v_theta_0 = cartesian_to_polar(x_0, y_0, vx_0, vy_0)

        # randomly vary final state
        a_f = self.np_random.uniform(low=a_range[0], high=a_range[1])
        e_f = self.np_random.uniform(low=e_range[0], high=e_range[1])
        w_f = self.np_random.uniform(low=w_range[0], high=w_range[1])
        theta_f = self.np_random.uniform(low=theta_range[0], high=theta_range[1])

        # convert final state to cartesian coordinates
        x_f, y_f, vx_f, vy_f = calc_cart_from_OE(a_f, e_f, w_f, theta_f, mu)

        # set the location of the central body
        x_cb = 0.0
        y_cb = 0.0
        vx_cb = 0.0
        vy_cb = 0.0

        # print(f"Env Reset: theta = {theta} rad" )

        # set the initial state of the environment
        self._state = np.array([x_0, y_0, vx_0, vy_0, mass, mu, x_f, y_f, vx_f, vy_f], dtype=np.float32)

        # set the location of the central body
        self._arr_cb = np.array([x_cb, y_cb, vx_cb, vy_cb], dtype=np.float32)

        # Initialize a spacecraft object with the state of the environment
        sc = Spacecraft(r_0, theta_0, r_dot_0, v_theta_0, mass, C1, C2)

        # Update the spacecraft in the environment
        self._spacecraft = sc

        # calculate orbital elements
        a, e, w, theta = sc.calc_Planar_OE(x_cb, y_cb, vx_cb, vy_cb, mu)

        self._keplerian_elements[0] = a
        self._keplerian_elements[1] = e
        self._keplerian_elements[2] = w
        self._keplerian_elements[3] = theta

        # non dim state in observation
        x_nd = x_0 / self.l_star
        y_nd = y_0 / self.l_star
        vx_nd = vx_0 / (self.l_star / self.t_star)
        vy_nd = vy_0 / (self.l_star / self.t_star)
        mass_nd = mass / self.m_star
        x_target_nd = x_f / self.l_star
        y_target_nd = y_f / self.l_star
        vx_target_nd = vx_f / (self.l_star / self.t_star)
        vy_target_nd = vy_f / (self.l_star / self.t_star)

        observation = np.array([x_nd, y_nd, vx_nd, vy_nd, mass_nd, x_target_nd, y_target_nd, vx_target_nd, vy_target_nd], dtype=np.float32)
        self.elapsed_t = 0.0

        info = self._get_info(
            None, #placeholder for ODE data - only provided in step()
            None, #placeholder for delta_r data - only provided in step()
        )

        return observation, info
    
    def calc_deltas(self):

        # unpack state vector
        x = self._state[0]
        y = self._state[1]
        vx = self._state[2]
        vy = self._state[3]
        mass = self._state[4]
        mu = self._state[5]
        x_target = self._state[6]
        y_target = self._state[7]
        vx_target = self._state[8]
        vy_target = self._state[9]

        # position difference from target
        dx = x - x_target    # delta x in m
        dy = y - y_target    # delta y in m
        dr = (dx**2 + dy**2) ** 0.5 # distance to target in m
        r_target = (x_target**2 + y_target**2) ** 0.5 # target radius in m
        dr_norm = dr / r_target if r_target != 0 else 0.0

        # velocity difference from target
        dvx = vx - vx_target    # delta vx in m/s
        dvy = vy - vy_target    # delta vy in m/s
        dv = (dvx**2 + dvy**2) ** 0.5    # velocity difference to target in m/s
        v_target = (vx_target**2 + vy_target**2) ** 0.5   # target velocity in m/s
        dv_norm = dv / v_target if v_target != 0 else 0.0

        return dx, dy, dr, dvx, dvy, dv, r_target, v_target, dr_norm, dv_norm

    def calc_reward(self):

        x = self._state[0]
        y = self._state[1]
        r = (x**2 + y**2) ** 0.5

        # extract orbital elements
        e = self._keplerian_elements[1]

        # determine reward based on state input, also check if state is terminal
        deltas = self.calc_deltas()
        dx, dy, dr, dvx, dvy, dv, r_target, v_target, dr_norm, dv_norm = deltas

        # central body parameters
        cb_rad = self.planet_radii[0]

        terminated = False

        # Check if a collision has taken place, terminate with negative reward
        # Otherwise, compute a reward based on distance from target SMA
        if r < cb_rad:
            reward = 0.0
            terminated = True
        elif e >= 1.0:
            reward = 0.0
            terminated = True
        else:
            # exponential decaying reward based on the difference between target
            # and desired position
            # reward shaping based on distance and velocity differences
            reward = - (dr_norm + dv_norm)

        return reward, terminated

    def _apply_dV_in_VNB_frame(self, dV, X_i, Y_i, VX_i, VY_i):
        # determine the vel magnitude
        v_norm = np.sqrt(VX_i**2 + VY_i**2)

        # calculate the current dV vector
        v_vec = np.array([VX_i, VY_i]) / v_norm

        # Multiply the delta-V magnitude by the velocity unit vector
        dV_vec = dV * v_vec

        return dV_vec

    def step(self, action):
        # unpack the state vector
        x = self._state[0]
        y = self._state[1]
        vx = self._state[2]
        vy = self._state[3]
        mass = self._state[4]
        mu = self._state[5]

        # central body location
        x_cb = self._arr_cb[0]
        y_cb = self._arr_cb[1]
        vx_cb = self._arr_cb[2]
        vy_cb = self._arr_cb[3]

        # get the current spacecraft object container
        sc = self._spacecraft

        # unpack the action vector
        u = action[0]  # throttle control
        alpha_x = action[1]  # spacecraft thrust x-direction
        alpha_y = action[2]  # spacecraft thrust y-direction

        # step the spacecraft forward
        t_span = (0.0, self.step_size)
        y0 = np.array([x, y, vx, vy, mass])
        params = np.array(
            [
                self.param_mu,
                sc.max_thrust,
                sc.specific_impulse,
                Constants.G0,
                u,
                alpha_x,
                alpha_y,
            ],
            dtype=np.float32,
        )

        # solve ODE
        solution = solve_ivp(env_EOM_TBT_v2, t_span, y0, method="RK45", args=(params,))

        # extract the final state vector from ODE solution (last column in y)
        y_final = (solution.y[:, -1]).astype(np.float32)

        # change in state vector
        delta_r = y_final - y0

        # update the state and elapsed time
        self.elapsed_t = self.elapsed_t + self.step_size
        self._state = np.append(y_final, [mu, self._state[6], self._state[7], self._state[8], self._state[9]]).astype(np.float32)

        x_plus = y_final[0]
        y_plus = y_final[1]
        vx_plus = y_final[2]
        vy_plus = y_final[3]
        mass_plus = y_final[4]

        # update the spacecraft object
        sc.update_state_cartesian(*y_final)

        # update the environment spacecraft object
        self._spacecraft = sc

        # calculate the new orbital elements
        a, e, w, theta = sc.calc_Planar_OE(x_cb, y_cb, vx_cb, vy_cb, mu)

        self._keplerian_elements[0] = a
        self._keplerian_elements[1] = e
        self._keplerian_elements[2] = w
        self._keplerian_elements[3] = theta

        # determine reward and terminated status
        reward, terminated = self.calc_reward()

        # accumulate reward
        self.episode_reward += reward

        # the observation is just the state vector
        # non dim state in observation
        x_nd = x_plus / self.l_star
        y_nd = y_plus / self.l_star
        vx_nd = vx_plus / (self.l_star / self.t_star)
        vy_nd = vy_plus / (self.l_star / self.t_star)
        mass_nd = mass_plus / self.m_star
        observation = np.array([x_nd, y_nd, vx_nd, vy_nd, mass_nd], dtype=np.float32)

        # extract other environment information
        info = self._get_info(solution, delta_r)

        truncated = False

        if terminated or truncated:
            info["episode"] = {"r": float(self.episode_reward)}

        return observation, reward, terminated, truncated, info

    def set_state(self, state):

        # unpack the input state vector
        x_in = state[0]
        y_in = state[1]
        vx_in = state[2]
        vy_in = state[3]
        m_in = state[4]

        # set the environment state to the input state vector
        self._state = np.array(
            [x_in, y_in, vx_in, vy_in, m_in, self._state[5], self._state[6]],
            dtype=np.float32,
        )

        # update the spacecraft object in the environment
        self._spacecraft.update_state_cartesian(x_in, y_in, vx_in, vy_in, m_in)

        # central body location
        x_cb = self._arr_cb[0]
        y_cb = self._arr_cb[1]
        vx_cb = self._arr_cb[2]
        vy_cb = self._arr_cb[3]

        # calculate the new orbital elements
        a, e, w, theta = self._spacecraft.calc_Planar_OE(
            x_cb, y_cb, vx_cb, vy_cb, self._state[5]
        )

        self._keplerian_elements[0] = a
        self._keplerian_elements[1] = e
        self._keplerian_elements[2] = w
        self._keplerian_elements[3] = theta
