import numpy as np
import gymnasium as gym
import warnings
from typing import Optional
from scipy.integrate import solve_ivp
from constants.constants import Constants
from core.spacecraft import Spacecraft
from core.propagation import env_EOM_TBT_v2
from utils.state_vector_utils import convert_radial_velocity_to_cartesian, polar_to_cartesian, calc_cart_from_OE, cartesian_to_polar, convert_fpa_to_velocity_components

class TwoBodyRendezvous_Polar_Env(gym.Env):
    def __init__(self, **kwargs):
        # define limits of the state parameters
        low_array = np.full(
            14, -np.inf, dtype=np.float32
        )  # lower bounds for state space
        high_array = np.full(
            14, np.inf, dtype=np.float32
        )  # upper-bounds for state space

        # define the state space (in this case the observation is the state) 10 elements
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
        self.a_min_init_env_nd = kwargs.get("a_min_init_env_nd", Constants.SMA_VENUS)  # min semi-major axis for env [AU]
        self.a_max_init_env_nd = kwargs.get("a_max_init_env_nd", Constants.SMA_MARS)  # max semi-major axis for env [AU]
        self.e_min_init_env = kwargs.get("e_min_init_env", 0.0)  # min eccentricity for env
        self.e_max_init_env = kwargs.get("e_max_init_env", 0.5)  # max eccentricity for env
        self.w_min_init_env_rad = kwargs.get("w_min_init_env_deg", 0.0) * np.pi / 180  # min argument of periapsis for env [rad]
        self.w_max_init_env_rad = kwargs.get("w_max_init_env_deg", 360) * np.pi / 180  # max argument of periapsis for env [rad]
        self.theta_min_init_env_rad = kwargs.get("theta_min_init_env_deg", 0.0) * np.pi / 180  # min true anomaly for env [rad]
        self.theta_max_init_env_rad = kwargs.get("theta_max_init_env_deg", 360) * np.pi / 180  # max true anomaly for env [rad]
        self.a_min_final_env_nd = kwargs.get("a_min_final_env_nd", Constants.SMA_VENUS)  # min semi-major axis for env [AU]
        self.a_max_final_env_nd = kwargs.get("a_max_final_env_nd", Constants.SMA_MARS)  # max semi-major axis for env [AU]
        self.e_min_final_env = kwargs.get("e_min_final_env", 0.0)  # min eccentricity for env
        self.e_max_final_env = kwargs.get("e_max_final_env", 0.5)  # max eccentricity for env
        self.w_min_final_env_rad = kwargs.get("w_min_final_env_deg", 0.0) * np.pi / 180  # min argument of periapsis for env [rad]
        self.w_max_final_env_rad = kwargs.get("w_max_final_env_deg", 360) * np.pi / 180  # max argument of periapsis for env [rad]
        self.theta_min_final_env_rad = kwargs.get("theta_min_final_env_deg", 0.0) * np.pi / 180  # min true anomaly for env [rad]
        self.theta_max_final_env_rad = kwargs.get("theta_max_final_env_deg", 360) * np.pi / 180  # max true anomaly for env [rad]
        # Support both naming conventions for weights
        self.pos_weight = kwargs.get("pos_r_weight", kwargs.get("r_weight", 1.0))
        self.vel_weight = kwargs.get("vel_r_weight", kwargs.get("v_weight", 1.0))
        self.mass_weight = kwargs.get("mass_r_weight", kwargs.get("mass_weight", 1.0))
        self.tof_scale = kwargs.get("tof_scale", 1.0)
        self.r_dist_weight = kwargs.get("r_dist_weight", self.pos_weight)
        self.v_dist_weight = kwargs.get("v_dist_weight", self.vel_weight)
        self.success_threshold_pos = kwargs.get("success_threshold_pos", 0.01)  # 1% of characteristic length
        self.success_threshold_vel = kwargs.get("success_threshold_vel", 0.01)  # 1% of characteristic velocity
        self.terminal_bonus = kwargs.get("terminal_bonus", 100.0)  # Large bonus for precise rendezvous
        self.precision_mult = kwargs.get("precision_mult", 10.0)  # Small bonus for being within success thresholds
        self.terminal_bonus = kwargs.get("terminal_bonus", 100.0)  # Large bonus for precise rendezvous
        self.tof_weight = kwargs.get("tof_weight", 1.0)  # Weighting factor for time component of reward
        self.time_dist_weight = kwargs.get("time_dist_weight", 1.0)  # Weighting factor for time distribution
        self.arr_r_polar_nd = np.array([0.0, 0.0], dtype=np.float32)  # placeholder for polar position [r, theta]
        self.arr_v_polar_nd = np.array([0.0, 0.0], dtype=np.float32)  # placeholder for polar velocity [v_r, v_theta]
        self.arr_rf_polar_nd = np.array([0.0, 0.0], dtype=np.float32)  # placeholder for target polar position [r, theta]
        self.arr_vf_polar_nd = np.array([0.0, 0.0], dtype=np.float32)  # placeholder for target polar velocity [v_r, v_theta]
        self.h_nd_0 = 0.0  # placeholder for initial angular momentum
        self.h_nd_f = 0.0  # placeholder for final angular momentum
        self.cos_eta = 0.0  # placeholder for eta cosine component
        self.sin_eta = 0.0  # placeholder for eta sine component
        self.alpha_x = 0.0  # placeholder for thrust direction x component
        self.alpha_y = 0.0  # placeholder for thrust direction y component

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

        mu = self.param_mu

        #check periods
        a_nd = self._keplerian_elements[0]/Constants.SMA_EARTH
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            try:
                T_nd = 2 * np.pi * ( a_nd**3 / 1.0 ) ** 0.5
            except RuntimeWarning:
                T_nd = np.nan
                
        #calc deltas
        deltas = self.calc_deltas()
        dx, dy, dr, dvx, dvy, dv, r_target, v_target, dr_norm, dv_norm = deltas

        x_target = self._state[5]
        y_target = self._state[6]
        vx_target = self._state[7]
        vy_target = self._state[8]

        r_target, theta_target, r_dot_target, v_theta_target = cartesian_to_polar(x_target, y_target, vx_target, vy_target)

        # Initialize a spacecraft object with the state of the environment
        sc_target = Spacecraft(r_target, theta_target, r_dot_target, v_theta_target, 1000.0, self.C1, self.C2)
        sc_target.update_state_cartesian(x_target, y_target, vx_target, vy_target, 1000.0)

        # calculate orbital elements
        a_target, e_target, w_target, theta_target = sc_target.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, mu)
        a_target_nd = a_target / Constants.SMA_EARTH
        T_target_nd = 2 * np.pi * (a_target_nd**3 / 1.0) ** 0.5

        if np.isnan(T_target_nd):
            print("T_target is NaN")
            print("a_target_nd:", a_target_nd)
            print(f"a_target: {a_target}, mu: {mu}")
            print(f"x_target: {x_target}, y_target: {y_target}, vx_target: {vx_target}, vy_target: {vy_target}")
            print(f"r_target: {r_target}, theta_target: {theta_target}, r_dot_target: {r_dot_target}, v_theta_target: {v_theta_target}")
            raise ValueError("T_target is NaN")

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
            "orbital_period_nd": T_nd,
            "orbital_period_years": T_nd * self.t_star / Constants.YEARS_TO_SEC,
            "a_target": a_target/Constants.SMA_EARTH,
            "e_target": e_target,
            "w_target": np.rad2deg(w_target),
            "theta_target": np.rad2deg(theta_target),
            "orbital_period_target_nd": T_target_nd,
            "orbital_period_target_years": T_target_nd * self.t_star / Constants.YEARS_TO_SEC,
            "mu": self.param_mu,
            "step_count": self.step_count,
            "TOF": self.TOF,
            "TOF_nd": self.TOF_nd,
            "timesteps_in_TOF": self.timesteps_in_TOF,
            "pos_residual": self.pos_residual,
            "vel_residual": self.vel_residual,
            "tof_scale": self.tof_scale,
            "r_weight": self.pos_weight,
            "v_weight": self.vel_weight,
            "mass_weight": self.mass_weight,
            "pos_r_component": self.pos_r_component,
            "vel_r_component": self.vel_r_component,
            "mass_r_component": self.mass_r_component,
            "r_nd": self.arr_r_polar_nd[0],
            "eta_nd": self.arr_r_polar_nd[1],
            "v_r_nd": self.arr_v_polar_nd[0],
            "v_eta_nd": self.arr_v_polar_nd[1],
            "r_nd_f": self.arr_rf_polar_nd[0],
            "eta_nd_f": self.arr_rf_polar_nd[1],
            "v_r_nd_f": self.arr_vf_polar_nd[0],
            "v_eta_nd_f": self.arr_vf_polar_nd[1],
            "h_nd_0": self.h_nd_0,
            "h_nd_f": self.h_nd_f,
            "cos_eta": self.cos_eta,
            "sin_eta": self.sin_eta,
            "cos_eta_f": self.cos_eta_f,
            "sin_eta_f": self.sin_eta_f,
            "fpa_nd": self.fpa_nd,
            "fpa_nd_f": self.fpa_f_nd,
            "fpa_cos": self.cos_fpa,
            "fpa_sin": self.sin_fpa,
            "alpha_x": self.alpha_x,
            "alpha_y": self.alpha_y,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # reset cumulative reward
        self.episode_reward = 0.0
        self.step_count = 0

        # extract central body gravitational parameter
        mu = self.param_mu

        # set ranges for initial state parameters
        a_range = [self.a_min_init_env_nd, self.a_max_init_env_nd]  # initial radius range (m)
        e_range = [self.e_min_init_env, self.e_max_init_env]  # initial eccentricity range
        w_range = [self.w_min_init_env_rad, self.w_max_init_env_rad]  # initial argument of periapsis range (rad)
        theta_range = [self.theta_min_init_env_rad, self.theta_max_init_env_rad]  # initial true anomaly range (rad)

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

        a_range = [self.a_min_final_env_nd, self.a_max_final_env_nd]  # final radius range (m)
        e_range = [self.e_min_final_env, self.e_max_final_env]  # final eccentricity range
        w_range = [self.w_min_final_env_rad, self.w_max_final_env_rad]  # final argument of periapsis range (rad)
        theta_range = [self.theta_min_final_env_rad, self.theta_max_final_env_rad]  # final true anomaly range (rad)

        # randomly vary final state
        a_f = self.np_random.uniform(low=a_range[0], high=a_range[1])
        e_f = self.np_random.uniform(low=e_range[0], high=e_range[1])
        w_f = self.np_random.uniform(low=w_range[0], high=w_range[1])
        theta_f = self.np_random.uniform(low=theta_range[0], high=theta_range[1])


        # convert the final state to polar coordinates and cartesian
        x_f, y_f, vx_f, vy_f = calc_cart_from_OE(a_f, e_f, w_f, theta_f, mu)
        r_f, theta_f, r_dot_f, v_theta_f = cartesian_to_polar(x_f, y_f, vx_f, vy_f)

        # convert final state to cartesian coordinates
        x_f, y_f, vx_f, vy_f = calc_cart_from_OE(a_f, e_f, w_f, theta_f, mu)

        # set the location of the central body
        x_cb = 0.0
        y_cb = 0.0
        vx_cb = 0.0
        vy_cb = 0.0

        # set the location of the central body
        self._arr_cb = np.array([x_cb, y_cb, vx_cb, vy_cb], dtype=np.float32)

        # Initialize a spacecraft object with the state of the environment
        sc = Spacecraft(r_0, theta_0, r_dot_0, v_theta_0, mass, C1, C2)
        sc_final = Spacecraft(r_f, theta_f, r_dot_f, v_theta_f, mass, C1, C2)

        # Update the spacecraft in the environment
        self._spacecraft = sc

        # calculate orbital elements
        a, e, w, theta = sc.calc_Planar_OE(x_cb, y_cb, vx_cb, vy_cb, mu)
        a_f, e_f, w_f, theta_f = sc_final.calc_Planar_OE(x_cb, y_cb, vx_cb, vy_cb, mu)
        a_nd = a / Constants.SMA_EARTH
        a_f_nd = a_f / Constants.SMA_EARTH
        T_i = 2 * np.pi * ( a_nd**3 / 1.0 ) ** 0.5
        T_target = 2 * np.pi * ( a_f_nd**3 / 1.0 ) ** 0.5

        #determine time of flight
        TTG_dim = 0.0
        TTG_nd = 0.0

        if (T_target > T_i):
            TTG_nd = T_target
            TTG_dim = T_target * self.t_star
        else:
            TTG_nd = T_i 
            TTG_dim = T_i * self.t_star


        TTG_nd = TTG_nd * self.tof_scale
        TTG_dim = TTG_nd * self.t_star

        self.TOF = TTG_dim
        self.TOF_nd = TTG_nd

        self.timesteps_in_TOF = int(TTG_dim / self.step_size) + 1

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

        # --------------------------------------------------------------------------------------------
        # create polar observation
        polar_obs = self.create_polar_observation(x_nd, y_nd, vx_nd, vy_nd, 
                                                  x_target_nd, y_target_nd, vx_target_nd, 
                                                  vy_target_nd, mass_nd, TTG_nd)
   

        # set the initial state of the environment
        self._state = np.array([x_0, y_0, vx_0, vy_0, mass, x_f, y_f, vx_f, vy_f, TTG_dim], dtype=np.float32)
        observation = polar_obs
        self.elapsed_t = 0.0

        deltas = self.calc_deltas()
        dx_nd, dy_nd, dr, dvx_nd, dvy_nd, dv_nd, r_target, v_target, dr_norm, dv_norm = deltas
        self.pos_residual = ( dx_nd**2 + dy_nd**2 ) ** 0.5
        self.vel_residual = ( dvx_nd**2 + dvy_nd**2 ) ** 0.5
        self.pos_r_component = 0.0
        self.vel_r_component = 0.0
        self.mass_r_component = 0.0

        info = self._get_info(
            None, #placeholder for ODE data - only provided in step()
            None, #placeholder for delta_r data - only provided in step()
        )

        return observation, info
    
    def create_polar_observation( self, x_nd, y_nd, vx_nd, vy_nd, 
                                 x_target_nd, y_target_nd, vx_target_nd, 
                                 vy_target_nd, mass_nd, TTG_nd ):

        # convert init nd state to polar
        r_nd_0, eta_nd_0, v_r_nd_0, v_eta_nd_0 = cartesian_to_polar(x_nd, y_nd, vx_nd, vy_nd)
        r_nd_f, eta_nd_f, v_r_nd_f, v_eta_nd_f = cartesian_to_polar(x_target_nd, y_target_nd, vx_target_nd, vy_target_nd)
        self.arr_r_polar_nd = [r_nd_0, eta_nd_0]
        self.arr_v_polar_nd = [v_r_nd_0, v_eta_nd_0]
        self.arr_rf_polar_nd = [r_nd_f, eta_nd_f]
        self.arr_vf_polar_nd = [v_r_nd_f, v_eta_nd_f]

        #determine angular momentum
        h_nd_0 = r_nd_0 * v_eta_nd_0
        h_nd_f = r_nd_f * v_eta_nd_f
        self.h_nd_0 = h_nd_0
        self.h_nd_f = h_nd_f

        #calculate eta trig components
        self.cos_eta = np.cos(eta_nd_0)
        self.sin_eta = np.sin(eta_nd_0)
        self.cos_eta_f = np.cos(eta_nd_f)
        self.sin_eta_f = np.sin(eta_nd_f)

        # total velocity magnitudes
        v_comp = ( vx_nd**2 + vy_nd**2 ) ** 0.5
        v_comp_f = ( vx_target_nd**2 + vy_target_nd**2 ) ** 0.5

        # transpose velocites
        self.v_t_nd = h_nd_0 / r_nd_0
        self.v_t_f_nd = h_nd_f / r_nd_f

        # radial velocities
        self.v_r_nd = ( max(v_comp**2 - self.v_t_nd**2, 1e-6) ) ** 0.5
        self.v_r_f_nd = ( max(v_comp_f**2 - self.v_t_f_nd**2, 1e-6) ) ** 0.5

        # flight path angles
        self.fpa_nd = np.arctan2( self.v_r_nd, self.v_t_nd )
        self.fpa_f_nd = np.arctan2( self.v_r_f_nd, self.v_t_f_nd )

        # flight path angle trig components
        self.cos_fpa = np.cos( self.fpa_nd )
        self.sin_fpa = np.sin( self.fpa_nd )
        self.cos_fpa_f = np.cos( self.fpa_f_nd )
        self.sin_fpa_f = np.sin( self.fpa_f_nd )

        #construct polar observation array
        polar_observation = np.array( [
            r_nd_0,
            self.cos_eta,
            self.sin_eta,
            v_comp,
            self.cos_fpa,
            self.sin_fpa,
            mass_nd,
            r_nd_f,
            self.cos_eta_f,
            self.sin_eta_f,
            v_comp_f,
            self.cos_fpa_f,
            self.sin_fpa_f,
            TTG_nd
            ],
            dtype=np.float32
         )

        return polar_observation
        

    def calc_deltas(self):

        # unpack state vector
        x = self._state[0]
        y = self._state[1]
        vx = self._state[2]
        vy = self._state[3]
        mass = self._state[4]
        x_target = self._state[5]
        y_target = self._state[6]
        vx_target = self._state[7]
        vy_target = self._state[8]

        # normalize target velocities
        x_target_norm = x_target / self.l_star
        y_target_norm = y_target / self.l_star
        vx_target_norm = vx_target / (self.l_star / self.t_star)
        vy_target_norm = vy_target / (self.l_star / self.t_star)

        # position difference from target
        dx = (x - x_target) / self.l_star   # delta x nd
        dy = (y - y_target) / self.l_star    # delta y nd
        dr = (dx**2 + dy**2) ** 0.5 # distance to target in nd units
        r_target = (x_target_norm**2 + y_target_norm**2) ** 0.5 # target radius in nd units
        dr_norm = dr / r_target if r_target != 0 else 0.0

        # velocity difference from target
        dvx = (vx - vx_target) / (self.l_star / self.t_star)  # delta vx in nd units
        dvy = (vy - vy_target) / (self.l_star / self.t_star)  # delta vy in nd units
        dv = (dvx**2 + dvy**2) ** 0.5    # velocity difference to target in nd units
        v_target = (vx_target_norm**2 + vy_target_norm**2) ** 0.5   # target velocity in nd units
        dv_norm = dv / v_target if v_target != 0 else 0.0

        return dx, dy, dr, dvx, dvy, dv, r_target, v_target, dr_norm, dv_norm

    def calc_reward(self):

        x = self._state[0] / self.l_star
        y = self._state[1] / self.l_star
        vx = self._state[2] / (self.l_star / self.t_star)
        vy = self._state[3] / (self.l_star / self.t_star)
        x_target = self._state[5] / self.l_star
        y_target = self._state[6] / self.l_star
        vx_target = self._state[7] / (self.l_star / self.t_star)
        vy_target = self._state[8] / (self.l_star / self.t_star)
        TTG = self._state[9]
        TTG_nd = TTG / self.t_star
        r = (x**2 + y**2) ** 0.5

        mass = self._state[4] / self.m_star

        # extract orbital elements
        e = self._keplerian_elements[1]

        # determine reward based on state input, also check if state is terminal
        deltas = self.calc_deltas()
        dx_nd, dy_nd, dr, dvx_nd, dvy_nd, dv_nd, r_target, v_target, dr_norm, dv_norm = deltas
        self.pos_residual = ( dx_nd**2 + dy_nd**2 ) ** 0.5
        self.vel_residual = ( dvx_nd**2 + dvy_nd**2 ) ** 0.5

        residual = dx_nd**2 + dy_nd**2 + dvx_nd**2 + dvy_nd**2

        # Separate exponentials for position and velocity - provides smoother gradient
        self.time_component = np.exp(- self.time_dist_weight * TTG_nd**2) * self.tof_weight
        self.pos_r_component = np.exp(- self.r_dist_weight * self.pos_residual**2) * self.pos_weight
        self.vel_r_component = np.exp(- self.v_dist_weight * self.vel_residual**2) * self.vel_weight
        self.mass_r_component = -self.mass_increment / self.m_star * self.mass_weight

        #Step-based shaping reward (always provided for learning)
        if self.pos_residual < self.success_threshold_pos and self.vel_residual < self.success_threshold_vel:
            precision_mult = self.precision_mult  # Small bonus for being within success thresholds
            self.pos_r_component = self.pos_r_component * precision_mult
            self.vel_r_component = self.vel_r_component * precision_mult

        #shaping_reward = self.time_component * ( self.pos_r_component + self.vel_r_component ) + self.mass_r_component
        self.pos_r_component = self.pos_r_component * self.time_component
        self.vel_r_component = self.vel_r_component * self.time_component
        shaping_reward = self.pos_r_component + self.vel_r_component + self.mass_r_component

        # central body parameters
        cb_rad = self.planet_radii[0] / self.l_star  # central body radius in nd units

        terminated = False

        
        if r < cb_rad:
            reward = 0.0
            terminated = True
        elif e >= 1.0:
            reward = 0.0
            #terminated = True
        elif TTG <= 0.0:          
            if self.pos_residual < self.success_threshold_pos and self.vel_residual < self.success_threshold_vel:
                terminal_bonus = self.terminal_bonus  # Large bonus for precise rendezvous
            else:
                terminal_bonus = 0.0
            
            reward = shaping_reward + terminal_bonus
            terminated = True
        else:
            # During trajectory: only shaping reward
            reward = shaping_reward

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
        TTG = self._state[9]
        mu = self.param_mu  # solar mu [m^3/s^2]
        self.step_count += 1

        # central body location
        x_cb = self._arr_cb[0]
        y_cb = self._arr_cb[1]
        vx_cb = self._arr_cb[2]
        vy_cb = self._arr_cb[3]

        # get the current spacecraft object container
        sc = self._spacecraft

        # unpack the action vector
        u = action[0]  # throttle control
        fpa_cos = action[1]
        fpa_sin = action[2]

        # get the flight path angle
        fpa = np.arctan2( fpa_sin, fpa_cos )
        
        # get radial and transverse unit vectors
        v_norm = np.sqrt(vx**2 + vy**2)
        vec_r, v_eta = convert_fpa_to_velocity_components( v_norm, fpa )

        # convert polar velocity components to cartesian
        eta = self.arr_r_polar_nd[1]
        alpha_x, alpha_y = convert_radial_velocity_to_cartesian( vec_r, v_eta, eta )
        alpha = ( alpha_x**2 + alpha_y**2 ) ** 0.5
        alpha_x = alpha_x / alpha if alpha != 0 else 0.0
        alpha_y = alpha_y / alpha if alpha != 0 else 0.0
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

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

        # solve ODE - catch step size warning as exception
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='Required step size is less than spacing between numbers')
            try:
                solution = solve_ivp(env_EOM_TBT_v2, t_span, y0, method="RK45", args=(params,))
            except UserWarning as w:
                raise Exception(f"Integration failed in env.step(): {str(w)}") from w

        # extract the final state vector from ODE solution (last column in y)
        y_final = (solution.y[:, -1]).astype(np.float32)

        # change in state vector
        delta_r = y_final - y0

        # update time to go
        TTG = TTG - self.step_size

        # update the state and elapsed time
        self.elapsed_t = self.elapsed_t + self.step_size
        self._state = np.append(y_final, [self._state[5], self._state[6], self._state[7], self._state[8], TTG]).astype(np.float32)

        x_plus = y_final[0]
        y_plus = y_final[1]
        vx_plus = y_final[2]
        vy_plus = y_final[3]
        mass_plus = y_final[4]

        self.mass_increment = mass - mass_plus

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
        x_target_nd = self._state[5] / self.l_star
        y_target_nd = self._state[6] / self.l_star
        vx_target_nd = self._state[7] / (self.l_star / self.t_star)
        vy_target_nd = self._state[8] / (self.l_star / self.t_star)
        TTG_nd = self._state[9] / self.t_star

        # --------------------------------------------------------------------------------------------
        # create polar observation
        polar_obs = self.create_polar_observation(x_nd, y_nd, vx_nd, vy_nd, 
                                                  x_target_nd, y_target_nd, vx_target_nd, 
                                                  vy_target_nd, mass_nd, TTG_nd)
        
        observation = polar_obs

        # extract other environment information
        info = self._get_info(solution, delta_r)

        truncated = False

        if terminated or truncated:
            info["episode"] = {"r": float(self.episode_reward)}

        return observation, reward, terminated, truncated, info
    
    def _convert_state_to_obs(self):

        x_nd = self._state[0] / self.l_star
        y_nd = self._state[1] / self.l_star
        vx_nd = self._state[2] / (self.l_star / self.t_star)
        vy_nd = self._state[3] / (self.l_star / self.t_star)
        mass_nd = self._state[4] / self.m_star
        x_target_nd = self._state[5] / self.l_star
        y_target_nd = self._state[6] / self.l_star
        vx_target_nd = self._state[7] / (self.l_star / self.t_star)
        vy_target_nd = self._state[8] / (self.l_star / self.t_star)
        TTG_nd = self._state[9] / self.t_star

        observation = np.array([x_nd, y_nd, vx_nd, vy_nd, mass_nd, x_target_nd, 
                                y_target_nd, vx_target_nd, vy_target_nd, TTG_nd], 
                                dtype=np.float32)
        
        return observation

    def set_state(self, state):

        # unpack the input state vector
        x_in = state[0]
        y_in = state[1]
        vx_in = state[2]
        vy_in = state[3]
        m_in = state[4]

        x_target_in = state[5]
        y_target_in = state[6]
        vx_target_in = state[7]
        vy_target_in = state[8]
        TTG_in = state[9]

        # set the environment state to the input state vector
        self._state = np.array(
            [x_in, y_in, vx_in, vy_in, m_in, x_target_in, y_target_in, vx_target_in, vy_target_in, TTG_in],
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
            x_cb, y_cb, vx_cb, vy_cb, self.param_mu
        )

        self._keplerian_elements[0] = a
        self._keplerian_elements[1] = e
        self._keplerian_elements[2] = w
        self._keplerian_elements[3] = theta

        # convert initial and final states to polar coordinates
        r_0, theta_0, r_dot_0, v_theta_0 = cartesian_to_polar(x_in, y_in, vx_in, vy_in)
        r_f, theta_f, r_dot_f, v_theta_f = cartesian_to_polar(x_target_in, y_target_in, vx_target_in, vy_target_in)

        # Initialize spacecraft objects for initial and final states
        sc_init = Spacecraft(r_0, theta_0, r_dot_0, v_theta_0, m_in, self.C1, self.C2)
        sc_fin = Spacecraft(r_f, theta_f, r_dot_f, v_theta_f, m_in, self.C1, self.C2)

        # calculate orbital elements for initial and final states
        a_init, e_init, w_init, theta_init = sc_init.calc_Planar_OE(
            x_cb, y_cb, vx_cb, vy_cb, self.param_mu
        )
        a_fin, e_fin, w_fin, theta_fin = sc_fin.calc_Planar_OE(
            x_cb, y_cb, vx_cb, vy_cb, self.param_mu
        )

        # determine orbital periods
        a_nd = a_init / Constants.SMA_EARTH
        a_f_nd = a_fin / Constants.SMA_EARTH
        T_i = 2 * np.pi * ( a_nd**3 / 1.0 ) ** 0.5
        T_target = 2 * np.pi * ( a_f_nd**3 / 1.0 ) ** 0.5

        #set mass increment to zero
        self.mass_increment = 0.0

        #determine total time of flight based on initial + target orbit
        if (T_target > T_i):
            self.TOF = T_target * self.t_star * self.tof_scale
            self.TOF_nd = T_target * self.tof_scale
        else:
            self.TOF = T_i * self.t_star * self.tof_scale
            self.TOF_nd = T_i * self.tof_scale

        self.timesteps_in_TOF = int(self.TOF / self.step_size) + 1

        info = self._get_info(
            None, #placeholder for ODE data - only provided in step()
            None, #placeholder for delta_r data - only provided in step()
        )

        # --------------------------------------------------------------------------------------------
        # create polar observation
        x_nd = x_in / self.l_star
        y_nd = y_in / self.l_star
        vx_nd = vx_in / (self.l_star / self.t_star)
        vy_nd = vy_in / (self.l_star / self.t_star)
        mass_nd = m_in / self.m_star
        x_target_nd = x_target_in / self.l_star
        y_target_nd = y_target_in / self.l_star
        vx_target_nd = vx_target_in / (self.l_star / self.t_star)
        vy_target_nd = vy_target_in / (self.l_star / self.t_star)
        TTG_nd = TTG_in / self.t_star

        polar_obs = self.create_polar_observation(x_nd, y_nd, vx_nd, vy_nd, 
                                                  x_target_nd, y_target_nd, vx_target_nd, 
                                                  vy_target_nd, mass_nd, TTG_nd)

        return polar_obs, info
    
    def get_cartesian_state(self):
        return self._state.copy()