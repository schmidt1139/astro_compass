import numpy as np
import gymnasium as gym
from typing import Optional
from scipy.integrate import solve_ivp
from constants.constants import Constants
from core.spacecraft import Spacecraft
from core.propagation import env_EOM_TBT_v2
from utils.state_vector_utils import compute_kep_velocities, orbit_equation, cartesian_to_polar, calc_cart_from_OE, convert_attitude_from_radial_to_cartesian, compute_eccentric_anomaly_from_true_anomaly, compute_mean_anomaly_from_eccentric_anomaly, calc_OE_from_cart
from utils.rl_utils import compute_obs_fast_TBT, compute_reward_fast_TBT

class TwoBody_Orb2Orb_Transfer_Env_target(gym.Env):
    def __init__(self, **kwargs):
        # define limits of the state parameters
        low_array = np.full(
            10, -np.inf, dtype=np.float32
        )  # lower bounds for state space
        high_array = np.full(
            10, np.inf, dtype=np.float32
        )  # upper-bounds for state space

        # define the state space (in this case the observation is the state) 5x5
        self.observation_space = gym.spaces.Box(low=low_array, high=high_array)

        self._state = np.full(7, 0.0, dtype=np.float32)  # initialize state vector

        self._keplerian_elements = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        self._keplerian_elements_target = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

        # list of default environment parameters (Sun is the central body)
        self.mu = Constants.MU_SUN * 10**9  # in m^3/s^2
        self.max_T = 1.33 / 1000  # max spacecraft thrust (in kN)
        self.ISP = 3872.0  # spacecraft specific impulse (s)
        self.C1 = 1.33 / 1000  # Spacecraft max thrust (in kN)
        self.C2 = 3872.0  # Spacecraft specific impulse (s)
        self.l_star = Constants.SMA_EARTH  # characteristic length (m)
        self.t_star = (149598023000**3 / (Constants.MU_SUN * 10 ** (9)))  # characteristic time (s)
        self.m_star = 3366.0  # characteristic mass (kg)
        self.step_size = 86400  # environment step size (s)
        self.mass_penalty = 0.0  # mass penalty factor
        self.arr_mu = np.array([self.mu / 10**9])  # solar mu [km^3/s^2]
        self.planet_radii = np.array([Constants.RADIUS_SUN_M])  # solar radius [m]
        self.a_min_init_env_nd = 1.0
        self.a_max_init_env_nd = 1.5
        self.e_min_init_env = 0.001
        self.e_max_init_env = 0.001
        self.w_min_init_env_deg = 0.0
        self.w_max_init_env_deg = 360.0
        self.theta_min_init_env_deg = 0.0
        self.theta_max_init_env_deg = 360.0
        self.a_min_final_env_nd = 1.0
        self.a_max_final_env_nd = 1.5
        self.e_min_final_env = 0.001
        self.e_max_final_env = 0.001
        self.w_min_final_env_deg = 0.0
        self.w_max_final_env_deg = 360.0
        self.pos_r_weight = 1.0
        self.vel_r_weight = 1.0
        self.throttle_r_weight = 0.0
        self.tof_scale = 1.0
        self.r_dist_weight = 1.0
        self.v_dist_weight = 1.0
        self.prop_length_scale = 1.0
        self.TTG = 0.0
        self.mass_initial = 3366.0  # initial mass (kg)

        self.elapsed_t = 0.0
        self.episode_reward = 0.0
        self.reward_distance = 0.0
        self.reward_mass = 0.0
        self.dm_nd = 0.0
        self.position_res = np.nan
        self.velocity_res = np.nan
        self.terminated = False
        self.truncated = False
        self.alpha_x = 1.0
        self.alpha_y = 0.0

        # define the action space
        # The action space consists of three variables:
        #    1) a control throlle input (scaled from 0 to 1)
        #    2) a thrust direction x vector component (ranges -1 to 1)
        #    3) a thrust direction y vector component (ranges -1 to 1)
        low_array_action = np.array([0.0, -1.0, -1.0], dtype=np.float32)
        high_array_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low_array_action, high=high_array_action)

        # double check kwargs and assign any passed parameters
        for key in kwargs:
            if not hasattr(self, key):
                raise ValueError(f"Unexpected keyword argument: {key}")
            else:
                setattr(self, key, kwargs[key])

        # set derived parameters
        self.C1 = self.max_T
        self.C2 = self.ISP
        self.w_min_init_env_rad = np.deg2rad(self.w_min_init_env_deg)
        self.w_max_init_env_rad = np.deg2rad(self.w_max_init_env_deg)
        self.theta_min_init_env_rad = np.deg2rad(self.theta_min_init_env_deg)
        self.theta_max_init_env_rad = np.deg2rad(self.theta_max_init_env_deg)
        self.w_min_final_env_rad = np.deg2rad(self.w_min_final_env_deg)
        self.w_max_final_env_rad = np.deg2rad(self.w_max_final_env_deg)

    def seed(self, seed_in: Optional[int] = None):
        # set the random seed for the environment
        self.seed = seed_in

    def _get_info(self, ode_solution, delta_r):
        # to-do: add orbital elements as optional and append to output dictionary

        self.state_a_nd = self._keplerian_elements[0]/self.l_star
        self.state_e_nd = self._keplerian_elements[1]
        self.state_w_deg = np.rad2deg(self._keplerian_elements[2])
        self.state_theta_deg = np.rad2deg(self._keplerian_elements[3])
        self.state_aol_deg = np.rad2deg(self.aol_0)

        self.target_a_nd = self._keplerian_elements_target[0]/self.l_star
        self.target_e_nd = self._keplerian_elements_target[1]
        self.target_w_deg = np.rad2deg(self._keplerian_elements_target[2])
        self.target_theta_deg = np.rad2deg(self._keplerian_elements_target[3])
        self.target_aol_deg = np.rad2deg(self.aol_target)

        self.orbital_period_days = self.T_i / 86400
        self.orbital_period_yrs = self.T_i / 86400 / 365.25
        self.target_period_days = self.T_target / 86400
        self.target_period_yrs = self.T_target / 86400 / 365.25

        eta_check = np.atan2( self.y_0, self.x_0 )
        eta_check_deg = np.rad2deg(eta_check) % 360

        #compute eccentric anomaly
        self.eccentric_anomaly = compute_eccentric_anomaly_from_true_anomaly(self._keplerian_elements[3], self._keplerian_elements[1])

        #compute mean anomaly
        self.mean_anomaly = compute_mean_anomaly_from_eccentric_anomaly(self.eccentric_anomaly, self._keplerian_elements[1])


        return {
            "Elapsed time": self.elapsed_t,
            "step_size_s": self.step_size,
            "step_size_days": self.step_size / 86400,
            "step_size_yrs": self.step_size / 86400 / 365.25,
            "mu": self.arr_mu[0],
            "ODE Solution": ode_solution,
            "delta_state": delta_r,
            "planet_radii": self.planet_radii,
            "a": self._keplerian_elements[0],
            "e": self._keplerian_elements[1],
            "w": self._keplerian_elements[2],
            "theta": self._keplerian_elements[3],
            "aol": self.aol_0,
            "state_x_nd": self.x_0 / self.l_star,
            "state_y_nd": self.y_0 / self.l_star,
            "state_vx_nd": self.vx_0 * self.t_star / self.l_star,
            "state_vy_nd": self.vy_0 * self.t_star / self.l_star,
            "state_a_nd": self.state_a_nd,
            "state_e_nd": self.state_e_nd,
            "state_w_deg": self.state_w_deg,
            "state_theta_deg": self.state_theta_deg,
            "state_aol_deg": self.state_aol_deg,
            "target_a_nd": self.target_a_nd,
            "target_e_nd": self.target_e_nd,
            "target_w_deg": self.target_w_deg,
            "target_theta_deg": self.target_theta_deg,
            "target_aol_deg": self.target_aol_deg,
            "eta_check_deg": eta_check_deg,
            "orbital_period_sec": self.T_i,
            "orbital_period_days": self.orbital_period_days,
            "orbital_period_years": self.orbital_period_yrs,
            "orbital_period_nd": self.T_i / self.t_star,
            "target_period_sec": self.T_target,
            "target_period_days": self.target_period_days,
            "target_period_years": self.target_period_yrs,
            "orbital_period_target_nd": self.T_target / self.t_star,
            "state_r_nd": self.r_0 / self.l_star,
            "state_eta_deg": np.rad2deg(self.eta_0),
            "state_r_dot_nd": self.r_dot_0 * self.t_star / self.l_star,
            "state_v_eta_nd": self.v_eta_0 * self.t_star / self.l_star,
            "state_aol_deg": np.rad2deg(self.aol_0),
            "max_thrust": self._spacecraft.max_thrust,
            "ISP": self._spacecraft.specific_impulse,
            "dm_nd": self.dm_nd,
            "reward_distance_component": self.reward_distance,
            "reward_mass_component": self.reward_mass,
            "state_eccentric_anomaly_deg": np.rad2deg(self.eccentric_anomaly),
            "state_mean_anomaly_deg": np.rad2deg(self.mean_anomaly),
            "position_res": self.position_res,
            "velocity_res": self.velocity_res,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "alpha_x": self.alpha_x,
            "alpha_y": self.alpha_y,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # reset cumulative reward
        self.episode_reward = 0.0

        # extract central body gravitational parameter
        mu = self.mu

        # set ranges for initial state parameters
        a_range = [
            self.a_min_init_env_nd,
            self.a_max_init_env_nd,
        ]  # initial radius range (m)
        e_range = [
            self.e_min_init_env,
            self.e_max_init_env,
        ]  # initial eccentricity range
        w_range = [
            self.w_min_init_env_rad,
            self.w_max_init_env_rad,
        ]  # initial argument of periapsis range (rad)
        theta_range = [
            self.theta_min_init_env_rad,
            self.theta_max_init_env_rad,
        ]  # initial true anomaly range (rad)

        # randomly vary initial state
        a_0 = self.np_random.uniform(low=a_range[0], high=a_range[1])
        e_0 = self.np_random.uniform(low=e_range[0], high=e_range[1])
        w_0 = self.np_random.uniform(low=w_range[0], high=w_range[1])
        theta_0 = self.np_random.uniform(low=theta_range[0], high=theta_range[1])
        aol_0 = (w_0 + theta_0) % (2 * np.pi)  #initial argument of latitude

        #set ranges for target orbit
        a_target_range = [
            self.a_min_final_env_nd,
            self.a_max_final_env_nd,
        ]  # target radius range (m)

        e_target_range = [
            self.e_min_final_env,
            self.e_max_final_env,
        ]  # target eccentricity range

        w_target_range = [
            self.w_min_final_env_rad,
            self.w_max_final_env_rad,
        ]  # target argument of periapsis range (rad)

        # randomly vary final state
        a_target = self.np_random.uniform(low=a_target_range[0], high=a_target_range[1])
        e_target = self.np_random.uniform(low=e_target_range[0], high=e_target_range[1])
        w_target = self.np_random.uniform(low=w_target_range[0], high=w_target_range[1])

        x_0, y_0, vx_0, vy_0 = calc_cart_from_OE(
            a_0,
            e_0,
            w_0,
            theta_0,
            mu,
        )

        # update env
        self.update_env(x_0, y_0, vx_0, vy_0, self.mass_initial, a_target, e_target, w_target)

        # RESET ONLY ----------------------------------------------------------------------
        if self.T_target > self.T_i:
            TTG_nd = self.T_target / self.t_star
            TTG_dim = self.T_target
        else:
            TTG_nd = self.T_i / self.t_star
            TTG_dim = self.T_i

        #scale the time of flight
        TTG_nd = TTG_nd * self.tof_scale
        TTG_dim = TTG_nd * self.t_star
        self.TTG = TTG_dim

        self.TTG_nd = TTG_nd
        self.TTG_dim = TTG_dim

        self.prop_duration = TTG_dim * self.prop_length_scale
        self.timesteps_in_TOF = int(TTG_dim / self.step_size) + 1
        self.timesteps_in_prop = self.timesteps_in_TOF * self.prop_length_scale

        params_temp = {
            "l_star": self.l_star,
            "t_star": self.t_star,
            "m_star": self.m_star,
        }
        observation, env_data = compute_obs_fast_TBT(self._state, params_temp, self.TTG_dim)

        for key in env_data:
            setattr(self, key, env_data[key])

        self.elapsed_t = 0.0

        info = self._get_info(
            None,
            None,
        )

        return observation, info

    def calc_reward(self, u):
        # determine reward based on state input, also check if state is terminal

        e = self._keplerian_elements[1]

        params_temp = {
            "l_star": self.l_star,
            "t_star": self.t_star,
            "m_star": self.m_star,
            "pos_r_weight": self.pos_r_weight,
            "vel_r_weight": self.vel_r_weight,
            "throttle_r_weight": self.throttle_r_weight,
            "tof_scale": self.tof_scale,
            "r_dist_weight": self.r_dist_weight,
            "v_dist_weight": self.v_dist_weight,
            "e": e,
            "max_T": self.max_T,
            "ISP": self.ISP,
        }
        reward, truncated, terminated, env_info = compute_reward_fast_TBT(self._state, params_temp, u, self.TTG)

        for key in env_info:
            setattr(self, key, env_info[key])

        return reward, terminated, truncated

    def step(self, action):
        # unpack the state vector
        x = self._state[0]
        y = self._state[1]
        vx = self._state[2]
        vy = self._state[3]
        mass = self._state[4]

        # central body location
        x_cb = self._arr_cb[0]
        y_cb = self._arr_cb[1]
        vx_cb = self._arr_cb[2]
        vy_cb = self._arr_cb[3]

        # get the current spacecraft object container
        sc = self._spacecraft

        # unpack the action vector
        u = action[0]  # throttle control
        alpha_r = action[1]  # radial direction component
        alpha_theta = action[2]  # transverse direction component

        # convert to x and y components
        alpha_x, alpha_y = convert_attitude_from_radial_to_cartesian(
            x, y, alpha_r, alpha_theta
        )

        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        # save non-dim mass
        mass_nd_0 = mass / self.m_star

        # step the spacecraft forward
        t_span = (0.0, self.step_size)
        x_km = x / 1000
        y_km = y / 1000
        vx_km_s = vx / 1000
        vy_km_s = vy / 1000
        mass_kg = mass
        mu_km3_s2 = self.mu * Constants.M3_TO_KM3
        y0 = np.array([x_km, y_km, vx_km_s, vy_km_s, mass_kg])
        params = np.array(
            [
                mu_km3_s2,
                sc.max_thrust,
                sc.specific_impulse,
                Constants.G0_KM,
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

        y_final_m = np.zeros(5, dtype=np.float32)
        y_final_m[0] = y_final[0] * 1000  # convert x back to m
        y_final_m[1] = y_final[1] * 1000  # convert y back to m
        y_final_m[2] = y_final[2] * 1000  # convert vx back to m/s
        y_final_m[3] = y_final[3] * 1000  # convert vy back to m/s
        y_final_m[4] = y_final[4]  # mass in kg

        x_0 = y_final_m[0]
        y_0 = y_final_m[1]
        vx_0 = y_final_m[2]
        vy_0 = y_final_m[3]
        mass_0 = y_final_m[4]

        # change in state vector
        delta_r = (y_final - y0) * np.array([1000, 1000, 1000, 1000, 1], dtype=np.float32)

        # update the state and elapsed time
        self.elapsed_t = self.elapsed_t + self.step_size
        
        # update the spacecraft object
        sc.update_state_cartesian(*y_final_m)

        # update the environment spacecraft object
        self._spacecraft = sc

        # update env
        self.update_env( x_0, y_0, vx_0, vy_0, mass_0, self.a_target, self.e_target, self.w_target )

        # update the time to go
        self.TTG -= self.step_size

        # determine reward and terminated status
        reward, terminated, truncated = self.calc_reward(u)

        # accumulate reward
        self.episode_reward += reward

        # construct observation
        params_temp = {
            "l_star": self.l_star,
            "t_star": self.t_star,
            "m_star": self.m_star,
        }
        observation, env_data = compute_obs_fast_TBT(self._state, params_temp, self.TTG)

        for key in env_data:
            setattr(self, key, env_data[key])

        # extract other environment information
        info = self._get_info(solution, delta_r)

        truncated = False

        if terminated or truncated:
            info["episode"] = {"r": float(self.episode_reward)}

        return observation, reward, terminated, truncated, info

    def set_state(self, state, ttg=None):

        if len(state) != 8:
            raise ValueError("Input state vector must have length 8.")

        # unpack the input state vector and convert m to km
        x_in = state[0]
        y_in = state[1]
        vx_in = state[2]
        vy_in = state[3]
        m_in = state[4]

        # input target orbital elements
        a_target_in = state[5]
        e_target_in = state[6]
        w_target_in = state[7]

        #time to go
        if ttg is None:
            TTG_in = self.TTG
        else:
            TTG_in = ttg
        self.TTG = TTG_in

        # update the env
        self.update_env(x_in, y_in, vx_in, vy_in, m_in, a_target_in, e_target_in, w_target_in)

        # construct observation
        params_temp = {
            "l_star": self.l_star,
            "t_star": self.t_star,
            "m_star": self.m_star,
        }
        observation, env_data = compute_obs_fast_TBT(self._state, params_temp, self.TTG)

        for key in env_data:
            setattr(self, key, env_data[key])

        info = self._get_info(
            None,
            None,
        )

        return observation, info

    def get_cartesian_state(self):
        return self._state.copy()
    
    def get_time_to_go(self):
        return self.TTG

    def update_env(self, x_0, y_0, vx_0, vy_0, mass_0, a_target, e_target, w_target):

        # calculate current orbital elements
        a_0, e_0, w_0, theta_0 = calc_OE_from_cart(x_0, y_0, vx_0, vy_0, self.mu)
        aol_0 = (w_0 + theta_0) % (2 * np.pi)  #initial argument of latitude
        
        # a_target, e_target, w_target are unchanged
        aol_target = aol_0

        # compute target true anomaly from argument of latitude
        theta_target = ( aol_target - w_target) % (2 * np.pi)

        # compute target orbit cartesian coordinates
        x_f, y_f, vx_f, vy_f = calc_cart_from_OE(a_target, e_target, w_target, theta_target, self.mu)

        # check that the argument of latitude matches
        aol_check = np.atan2( y_0, x_0 )
        if aol_check < 0:
            aol_check = aol_check + 2 * np.pi

        if (e_0 < 1.0):
            # Normalize both angles to [0, 2π)
            aol_0_norm = aol_0 % (2 * np.pi)
            aol_check_norm = aol_check % (2 * np.pi)
            
            # Compute the minimum angular difference accounting for wraparound
            angle_diff = abs(aol_check_norm - aol_0_norm)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            # Check orbital element consistency
            if angle_diff > 0.05:  # ~2.9 degrees tolerance - accounts for numerical precision near wraparound
                print(f"ERROR: w_0={np.rad2deg(w_0):.2f}°, theta_0={np.rad2deg(theta_0):.2f}°")
                print(f"ERROR: aol_check={np.rad2deg(aol_check_norm):.2f}°, aol_0={np.rad2deg(aol_0_norm):.2f}°")
                print(f"ERROR: Elapsed time: {self.elapsed_t}s")
                raise ValueError(f"Argument of latitude mismatch: aol_check={np.rad2deg(aol_check_norm):.2f}°, aol_0={np.rad2deg(aol_0_norm):.2f}°, diff={np.rad2deg(angle_diff):.2f}°")
        
        # compute polar coordinates
        r_0, eta_0, r_dot_0, v_eta_0 = cartesian_to_polar(x_0, y_0, vx_0, vy_0)
        r_f, eta_f, r_dot_f, v_eta_f = cartesian_to_polar(x_f, y_f, vx_f, vy_f)

        # set the location of the central body
        x_cb = 0.0
        y_cb = 0.0
        vx_cb = 0.0
        vy_cb = 0.0

        # set the initial state of the environment
        self._state = np.array([x_0, y_0, vx_0, vy_0, mass_0, x_f, y_f, vx_f, vy_f], dtype=np.float32)

        # set the location of the central body
        self._arr_cb = np.array([x_cb, y_cb, vx_cb, vy_cb], dtype=np.float32)

        # Initialize a spacecraft object with the state of the environment
        sc = Spacecraft(0.0, 0.0, 0.0, 0.0, mass_0, self.C1, self.C2)
        sc_target = Spacecraft(0.0, 0.0, 0.0, 0.0, mass_0, self.C1, self.C2)
        sc.update_state_cartesian(x_0, y_0, vx_0, vy_0, mass_0)
        sc_target.update_state_cartesian(x_f, y_f, vx_f, vy_f, mass_0)

        # Calculate time of flight scale - check for valid semi-major axis
        if a_0 >= 0:
            T_i = 2 * np.pi * (a_0**3 / self.mu ) ** 0.5
        else:
            T_i = np.nan

        if a_target >= 0:
            T_target = 2 * np.pi * (a_target**3 / self.mu) ** 0.5
        else:
            T_target = np.nan


        # update the state variables
        self.a_0 = a_0
        self.e_0 = e_0
        self.w_0 = w_0
        self.theta_0 = theta_0
        self.aol_0 = aol_0
        self.a_target = a_target
        self.e_target = e_target
        self.w_target = w_target
        self.theta_target = theta_target
        self.aol_target = aol_target
        self.x_0 = x_0
        self.y_0 = y_0
        self.vx_0 = vx_0
        self.vy_0 = vy_0
        self.m_0 = mass_0
        self.x_f = x_f
        self.y_f = y_f
        self.vx_f = vx_f
        self.vy_f = vy_f
        self.r_0 = r_0
        self.eta_0 = eta_0
        self.r_dot_0 = r_dot_0
        self.v_eta_0 = v_eta_0
        self.r_f = r_f
        self.eta_f = eta_f
        self.r_dot_f = r_dot_f
        self.v_eta_f = v_eta_f
        self.x_nd = x_0 / self.l_star
        self.y_nd = y_0 / self.l_star
        self.vx_nd = vx_0 / self.l_star * self.t_star
        self.vy_nd = vy_0 / self.l_star * self.t_star
        self.x_nd_target = x_f / self.l_star
        self.y_nd_target = y_f / self.l_star
        self.vx_nd_target = vx_f / self.l_star * self.t_star
        self.vy_nd_target = vy_f / self.l_star * self.t_star

        # calculate the initial orbital elements
        self._keplerian_elements[0] = a_0
        self._keplerian_elements[1] = e_0
        self._keplerian_elements[2] = w_0
        self._keplerian_elements[3] = theta_0

        self._keplerian_elements_target[0] = a_target
        self._keplerian_elements_target[1] = e_target
        self._keplerian_elements_target[2] = w_target
        self._keplerian_elements_target[3] = theta_target

        self.T_i = T_i
        self.T_target = T_target

        # Update the spacecraft in the environment
        self._spacecraft = sc
        self._spacecraft_target = sc_target
