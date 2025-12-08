"""Unified Two-Body rendezvous environment.

This merges the legacy `TwoBodyRendezvous_Polar_Env` (polar obs, FPA actions,
static target, legacy reward) and `TwoBodyRendezvous_Polar_Env2` (relative obs,
radial actions, propagated target, fast reward) behind a single configurable
class. Defaults mirror the Env2 behavior.
"""

import warnings
from typing import Literal, Optional

import gymnasium as gym
import numpy as np
from constants.constants import Constants
from core.propagation import env_EOM_TBT_v2
from core.spacecraft import Spacecraft
from scipy.integrate import solve_ivp
from utils.rl_utils import compute_reward_fast, create_relative_polar_observation_fast
from utils.state_vector_utils import (
    calc_cart_from_OE,
    cartesian_to_polar,
    convert_attitude_from_radial_to_cartesian,
    convert_fpa_to_velocity_components,
    convert_radial_velocity_to_cartesian,
)

TargetStrategy = Literal["propagated", "static"]
ActionMode = Literal["radial", "fpa"]
ObsMode = Literal["relative", "polar"]
RewardMode = Literal["throttle", "mass"]


class TwoBodyRendezvousEnv(gym.Env):
    """Unified rendezvous environment (env2 by default).

    Modes:
    - target_strategy: propagated (env2) | static (env1)
    - action_mode: radial (env2) | fpa (env1)
    - obs_mode: relative (env2 22-dim) | polar (env1 14-dim)
    - reward_mode: throttle (env2) | mass (env1)
    """

    def __init__(
        self,
        *,
        target_strategy: TargetStrategy = "propagated",
        action_mode: ActionMode = "radial",
        obs_mode: ObsMode = "relative",
        reward_mode: RewardMode = "throttle",
        **kwargs,
    ):
        super().__init__()

        self.target_strategy = target_strategy
        self.action_mode = action_mode
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode

        # define limits of the observation
        obs_dim = 22 if obs_mode == "relative" else 14
        low_array = np.full(obs_dim, -np.inf, dtype=np.float32)
        high_array = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low_array, high=high_array)

        # internal state
        self._state = np.full(10, 0.0, dtype=np.float32)
        self._keplerian_elements = np.zeros(6, dtype=np.float32)

        # list of environment parameters (Sun is the central body)
        self.param_mu = kwargs.get("mu", Constants.MU_SUN_M)  # in m^3/s^2
        self.C1 = kwargs.get("max_T", 1.33)  # Spacecraft max thrust (in N)
        self.C2 = kwargs.get("ISP", 3872.0)  # Spacecraft specific impulse (s)
        self.l_star = kwargs.get("l_star", Constants.SMA_EARTH)  # length (m)
        self.t_star = kwargs.get(
            "t_star", (Constants.SMA_EARTH**3 / (Constants.MU_SUN_M) ** 0.5)
        )  # time (s)
        self.m_star = kwargs.get("m_star", 3366.0)  # mass (kg)
        self.step_size = kwargs.get("step_size", 86400)  # environment step size (s)
        self.a_min_init_env_nd = kwargs.get("a_min_init_env_nd", Constants.SMA_VENUS)
        self.a_max_init_env_nd = kwargs.get("a_max_init_env_nd", Constants.SMA_MARS)
        self.e_min_init_env = kwargs.get("e_min_init_env", 0.0)
        self.e_max_init_env = kwargs.get("e_max_init_env", 0.5)
        self.w_min_init_env_rad = kwargs.get("w_min_init_env_deg", 0.0) * np.pi / 180
        self.w_max_init_env_rad = kwargs.get("w_max_init_env_deg", 360) * np.pi / 180
        self.theta_min_init_env_rad = (
            kwargs.get("theta_min_init_env_deg", 0.0) * np.pi / 180
        )
        self.theta_max_init_env_rad = (
            kwargs.get("theta_max_init_env_deg", 360) * np.pi / 180
        )
        self.a_min_final_env_nd = kwargs.get("a_min_final_env_nd", Constants.SMA_VENUS)
        self.a_max_final_env_nd = kwargs.get("a_max_final_env_nd", Constants.SMA_MARS)
        self.e_min_final_env = kwargs.get("e_min_final_env", 0.0)
        self.e_max_final_env = kwargs.get("e_max_final_env", 0.5)
        self.w_min_final_env_rad = kwargs.get("w_min_final_env_deg", 0.0) * np.pi / 180
        self.w_max_final_env_rad = kwargs.get("w_max_final_env_deg", 360) * np.pi / 180
        self.theta_min_final_env_rad = (
            kwargs.get("theta_min_final_env_deg", 0.0) * np.pi / 180
        )
        self.theta_max_final_env_rad = (
            kwargs.get("theta_max_final_env_deg", 360) * np.pi / 180
        )

        # reward weights
        self.pos_weight = kwargs.get("pos_r_weight", kwargs.get("r_weight", 1.0))
        self.vel_weight = kwargs.get("vel_r_weight", kwargs.get("v_weight", 1.0))
        self.mass_weight = kwargs.get("mass_r_weight", kwargs.get("mass_weight", 1.0))
        self.throttle_r_weight = kwargs.get(
            "throttle_r_weight", kwargs.get("mass_r_weight", 1.0)
        )
        self.tof_scale = kwargs.get("tof_scale", 1.0)
        self.r_dist_weight = kwargs.get("r_dist_weight", self.pos_weight)
        self.v_dist_weight = kwargs.get("v_dist_weight", self.vel_weight)
        self.success_threshold_pos = kwargs.get("success_threshold_pos", 0.01)
        self.success_threshold_vel = kwargs.get("success_threshold_vel", 0.01)
        self.terminal_bonus = kwargs.get("terminal_bonus", 100.0)
        self.precision_mult = kwargs.get("precision_mult", 10.0)
        self.tof_weight = kwargs.get("tof_weight", 1.0)
        self.time_dist_weight = kwargs.get("time_dist_weight", 1.0)
        self.prop_length_scale = kwargs.get("prop_length_scale", 1.0)
        self.max_episode_steps = kwargs.get("max_episode_steps", 5000)
        self.hyperbolic_termination = kwargs.get("hyperbolic_termination", False)

        self.arr_r_polar_nd = np.zeros(2, dtype=np.float32)
        self.arr_v_polar_nd = np.zeros(2, dtype=np.float32)
        self.arr_rf_polar_nd = np.zeros(2, dtype=np.float32)
        self.arr_vf_polar_nd = np.zeros(2, dtype=np.float32)

        self.arr_mu = np.array([self.param_mu])
        self.planet_radii = np.array([Constants.RADIUS_SUN_M])
        self.elapsed_t = 0.0
        self.episode_reward = 0.0
        self.mass_increment = 0.0
        self.terminated = False
        self.truncated = False

        # action space shared by both behaviors
        low_array_action = np.array([0.0, -1.0, -1.0], dtype=np.float32)
        high_array_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low_array_action, high=high_array_action)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _sample_state(self, mu):
        a_0 = self.np_random.uniform(
            low=self.a_min_init_env_nd, high=self.a_max_init_env_nd
        )
        e_0 = self.np_random.uniform(low=self.e_min_init_env, high=self.e_max_init_env)
        w_0 = self.np_random.uniform(
            low=self.w_min_init_env_rad, high=self.w_max_init_env_rad
        )
        theta_0 = self.np_random.uniform(
            low=self.theta_min_init_env_rad, high=self.theta_max_init_env_rad
        )
        x_0, y_0, vx_0, vy_0 = calc_cart_from_OE(a_0, e_0, w_0, theta_0, mu)

        a_f = self.np_random.uniform(
            low=self.a_min_final_env_nd, high=self.a_max_final_env_nd
        )
        e_f = self.np_random.uniform(
            low=self.e_min_final_env, high=self.e_max_final_env
        )
        w_f = self.np_random.uniform(
            low=self.w_min_final_env_rad, high=self.w_max_final_env_rad
        )
        theta_f = self.np_random.uniform(
            low=self.theta_min_final_env_rad, high=self.theta_max_final_env_rad
        )
        x_f, y_f, vx_f, vy_f = calc_cart_from_OE(a_f, e_f, w_f, theta_f, mu)
        return (x_0, y_0, vx_0, vy_0), (x_f, y_f, vx_f, vy_f)

    def _compute_tof(self, a, a_f):
        a_nd = a / Constants.SMA_EARTH
        a_f_nd = a_f / Constants.SMA_EARTH
        T_i = 2 * np.pi * (a_nd**3 / 1.0) ** 0.5
        T_target = 2 * np.pi * (a_f_nd**3 / 1.0) ** 0.5
        TTG_nd = T_target if T_target > T_i else T_i
        TTG_nd *= self.tof_scale
        TTG_dim = TTG_nd * self.t_star
        return TTG_nd, TTG_dim

    # ---------------------------------------------------------------------
    # reset
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.episode_reward = 0.0
        self.step_count = 0

        mu = self.param_mu

        (x_0, y_0, vx_0, vy_0), (x_f, y_f, vx_f, vy_f) = self._sample_state(mu)

        mass = 3366.0
        sc = Spacecraft(
            *cartesian_to_polar(x_0, y_0, vx_0, vy_0), mass, self.C1, self.C2
        )
        sc_target = Spacecraft(
            *cartesian_to_polar(x_f, y_f, vx_f, vy_f), mass, self.C1, self.C2
        )

        self._arr_cb = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._spacecraft = sc
        a, e, w, theta = sc.calc_Planar_OE(*self._arr_cb, mu)
        a_f, _, _, _ = sc_target.calc_Planar_OE(*self._arr_cb, mu)
        TTG_nd, TTG_dim = self._compute_tof(a, a_f)

        self.TOF = TTG_dim
        self.TOF_nd = TTG_nd
        self.timesteps_in_TOF = int(TTG_dim / self.step_size) + 1
        self.timesteps_in_prop = self.timesteps_in_TOF * self.prop_length_scale

        self._keplerian_elements[:4] = [a, e, w, theta]

        self._state = np.array(
            [x_0, y_0, vx_0, vy_0, mass, x_f, y_f, vx_f, vy_f, TTG_dim],
            dtype=np.float32,
        )
        self.mass_increment = 0.0

        # Target propagation
        target_state_t = (
            self.get_target_cartesian_state_at_ttg(TTG_dim)
            if self.target_strategy == "propagated"
            else np.array([x_f, y_f, vx_f, vy_f, mass], dtype=np.float32)
        )
        self.target_state_current = target_state_t

        # observation
        if self.obs_mode == "relative":
            params_temp = {
                "l_star": self.l_star,
                "t_star": self.t_star,
                "m_star": self.m_star,
                "mu": self.param_mu,
            }
            current_state_t = self._state[0:5]
            obs, env_info = create_relative_polar_observation_fast(
                params_temp, current_state_t, target_state_t, self._state[9]
            )
            for k, v in env_info.items():
                setattr(self, k, v)
            observation = obs
        else:
            observation = self.create_polar_observation(
                x_0 / self.l_star,
                y_0 / self.l_star,
                vx_0 / (self.l_star / self.t_star),
                vy_0 / (self.l_star / self.t_star),
                x_f / self.l_star,
                y_f / self.l_star,
                vx_f / (self.l_star / self.t_star),
                vy_f / (self.l_star / self.t_star),
                mass / self.m_star,
                TTG_nd,
            )

        self.pos_residual = 0.0
        self.vel_residual = 0.0
        self.pos_reward = 0.0
        self.vel_reward = 0.0
        self.mass_reward = 0.0
        self.throttle_reward = 0.0
        self.time_reward = 0.0

        info = self._get_info(None, None)
        return observation, info

    # ---------------------------------------------------------------------
    # observations
    def create_polar_observation(
        self,
        x_nd,
        y_nd,
        vx_nd,
        vy_nd,
        x_target_nd,
        y_target_nd,
        vx_target_nd,
        vy_target_nd,
        mass_nd,
        TTG_nd,
    ):
        r_nd_0, eta_nd_0, v_r_nd_0, v_eta_nd_0 = cartesian_to_polar(
            x_nd, y_nd, vx_nd, vy_nd
        )
        r_nd_target, eta_nd_target, v_r_nd_f, v_eta_nd_f = cartesian_to_polar(
            x_target_nd, y_target_nd, vx_target_nd, vy_target_nd
        )
        self.arr_r_polar_nd = np.array([r_nd_0, eta_nd_0], dtype=np.float32)
        self.arr_v_polar_nd = np.array([v_r_nd_0, v_eta_nd_0], dtype=np.float32)
        self.arr_rf_polar_nd = np.array([r_nd_target, eta_nd_target], dtype=np.float32)
        self.arr_vf_polar_nd = np.array([v_r_nd_f, v_eta_nd_f], dtype=np.float32)

        h_nd_0 = r_nd_0 * v_eta_nd_0
        h_nd_target = r_nd_target * v_eta_nd_f
        self.h_nd_0 = h_nd_0
        self.h_nd_f = h_nd_target

        self.cos_eta = np.cos(eta_nd_0)
        self.sin_eta = np.sin(eta_nd_0)
        self.cos_eta_f = np.cos(eta_nd_target)
        self.sin_eta_f = np.sin(eta_nd_target)

        v_comp = float(np.sqrt(vx_nd**2 + vy_nd**2))
        v_comp_target = float(np.sqrt(vx_target_nd**2 + vy_target_nd**2))

        self.v_t_nd = h_nd_0 / r_nd_0
        self.v_t_target_nd = h_nd_target / r_nd_target

        self.v_r_nd = float(np.sqrt(max(v_comp**2 - self.v_t_nd**2, 1e-6)))
        self.v_r_target_nd = float(
            np.sqrt(max(v_comp_target**2 - self.v_t_target_nd**2, 1e-6))
        )

        self.fpa_nd = np.arctan2(self.v_r_nd, self.v_t_nd)
        self.fpa_f_nd = np.arctan2(self.v_r_target_nd, self.v_t_target_nd)

        self.cos_fpa = np.cos(self.fpa_nd)
        self.sin_fpa = np.sin(self.fpa_nd)
        self.cos_fpa_f = np.cos(self.fpa_f_nd)
        self.sin_fpa_f = np.sin(self.fpa_f_nd)

        self.v_r_unit = self.v_r_nd / v_comp if v_comp != 0 else 0.0
        self.v_t_unit = self.v_t_nd / v_comp if v_comp != 0 else 0.0
        self.v_r_target_unit = (
            self.v_r_target_nd / v_comp_target if v_comp_target != 0 else 0.0
        )
        self.v_t_target_unit = (
            self.v_t_target_nd / v_comp_target if v_comp_target != 0 else 0.0
        )

        return np.array(
            [
                r_nd_0,
                self.cos_eta,
                self.sin_eta,
                v_comp,
                self.cos_fpa,
                self.sin_fpa,
                mass_nd,
                r_nd_target,
                self.cos_eta_f,
                self.sin_eta_f,
                v_comp_target,
                self.v_r_target_unit,
                self.v_t_target_unit,
                TTG_nd,
            ],
            dtype=np.float32,
        )

    def create_relative_polar_observation(
        self, x_nd, y_nd, vx_nd, vy_nd, mass_nd, TTG_nd
    ):
        params_temp = {
            "l_star": self.l_star,
            "t_star": self.t_star,
            "m_star": self.m_star,
        }
        TTG_dim = TTG_nd * self.t_star
        current_state_t = self._state[0:5]
        target_state_t = self.get_target_cartesian_state_at_ttg(TTG_dim)
        polar_observation, env_data = create_relative_polar_observation_fast(
            params_temp, current_state_t, target_state_t, TTG_dim
        )
        for key, value in env_data.items():
            setattr(self, key, value)
        return polar_observation

    # ---------------------------------------------------------------------
    def calc_deltas(self):
        x, y, vx, vy = self._state[0:4]
        x_target, y_target, vx_target, vy_target = self._state[5:9]

        x_target_norm = x_target / self.l_star
        y_target_norm = y_target / self.l_star
        vx_target_norm = vx_target / (self.l_star / self.t_star)
        vy_target_norm = vy_target / (self.l_star / self.t_star)

        dx = (x - x_target) / self.l_star
        dy = (y - y_target) / self.l_star
        dr = float(np.sqrt(dx**2 + dy**2))
        r_target = float(np.sqrt(x_target_norm**2 + y_target_norm**2))
        dr_norm = dr / r_target if r_target != 0 else 0.0

        dvx = (vx - vx_target) / (self.l_star / self.t_star)
        dvy = (vy - vy_target) / (self.l_star / self.t_star)
        dv = float(np.sqrt(dvx**2 + dvy**2))
        v_target = float(np.sqrt(vx_target_norm**2 + vy_target_norm**2))
        dv_norm = dv / v_target if v_target != 0 else 0.0
        return dx, dy, dr, dvx, dvy, dv, r_target, v_target, dr_norm, dv_norm

    def _calc_reward_mass(self):
        x_nd = self._state[0] / self.l_star
        y_nd = self._state[1] / self.l_star
        vx_nd = self._state[2] / (self.l_star / self.t_star)
        vy_nd = self._state[3] / (self.l_star / self.t_star)
        TTG = self._state[9]
        TTG_nd = TTG / self.t_star
        r = float(np.sqrt(x_nd**2 + y_nd**2))

        deltas = self.calc_deltas()
        dx_nd, dy_nd, _, dvx_nd, dvy_nd, _, _, _, _, _ = deltas
        self.pos_residual = float(np.sqrt(dx_nd**2 + dy_nd**2))
        self.vel_residual = float(np.sqrt(dvx_nd**2 + dvy_nd**2))

        time_component = np.exp(-self.time_dist_weight * TTG_nd**2) * self.tof_weight
        self.pos_reward = (
            np.exp(-self.r_dist_weight * self.pos_residual**2) * self.pos_weight
        )
        self.vel_reward = (
            np.exp(-self.v_dist_weight * self.vel_residual**2) * self.vel_weight
        )
        self.mass_reward = -self.mass_increment / self.m_star * self.mass_weight

        if (
            self.pos_residual < self.success_threshold_pos
            and self.vel_residual < self.success_threshold_vel
        ):
            self.pos_reward *= self.precision_mult
            self.vel_reward *= self.precision_mult

        self.pos_reward *= time_component
        self.vel_reward *= time_component
        shaping_reward = self.pos_reward + self.vel_reward + self.mass_reward

        cb_rad = self.planet_radii[0] / self.l_star
        terminated = False
        if r < cb_rad:
            reward = 0.0
            terminated = True
        elif self._keplerian_elements[1] >= 1.0 and self.hyperbolic_termination:
            reward = 0.0
            terminated = True
        elif TTG <= 0.0:
            terminal_bonus = (
                self.terminal_bonus
                if self.pos_residual < self.success_threshold_pos
                and self.vel_residual < self.success_threshold_vel
                else 0.0
            )
            reward = shaping_reward + terminal_bonus
            terminated = True
        else:
            reward = shaping_reward
        return float(reward), terminated, False

    def _calc_reward_throttle(self, u):
        params_temp = {
            "l_star": self.l_star,
            "t_star": self.t_star,
            "m_star": self.m_star,
            "max_T": self.C1,
            "ISP": self.C2,
            "mu": self.param_mu,
            "time_dist_weight": self.time_dist_weight,
            "tof_weight": self.tof_weight,
            "r_dist_weight": self.r_dist_weight,
            "v_dist_weight": self.v_dist_weight,
            "pos_r_weight": self.pos_weight,
            "vel_r_weight": self.vel_weight,
            "throttle_r_weight": self.throttle_r_weight,
            "success_threshold_pos": self.success_threshold_pos,
            "success_threshold_vel": self.success_threshold_vel,
            "precision_mult": self.precision_mult,
            "terminal_bonus": self.terminal_bonus,
        }
        current_state_t = self._state[0:5]
        ttg = self._state[9]
        target_state_t = self.get_target_cartesian_state_at_ttg(ttg)
        step_count = self.step_count
        total_timesteps = self.timesteps_in_prop

        reward, terminated, truncated, env_info = compute_reward_fast(
            params_temp,
            current_state_t,
            ttg,
            target_state_t,
            u,
            step_count,
            total_timesteps,
        )

        self.pos_residual = env_info.get("pos_residual", 0.0)
        self.vel_residual = env_info.get("vel_residual", 0.0)
        self.time_reward = env_info.get("time_r_component", 0.0)
        self.pos_reward = env_info.get("pos_r_component", 0.0)
        self.vel_reward = env_info.get("vel_r_component", 0.0)
        self.throttle_reward = env_info.get("throttle_r_component", 0.0)
        self.mass_reward = env_info.get("mass_r_component", 0.0)
        self.terminated = terminated
        return float(reward), bool(terminated), bool(truncated)

    # ---------------------------------------------------------------------
    def step(self, action):
        x, y, vx, vy, mass, x_t, y_t, vx_t, vy_t, TTG = self._state
        mu = self.param_mu
        self.step_count += 1

        sc = self._spacecraft

        # actions
        u = action[0]
        if self.action_mode == "radial":
            alpha_r = action[1]
            alpha_theta = action[2]
            alpha_x, alpha_y = convert_attitude_from_radial_to_cartesian(
                x, y, alpha_r, alpha_theta
            )
        else:
            fpa_cos = action[1]
            fpa_sin = action[2]
            fpa = np.arctan2(fpa_sin, fpa_cos)
            v_norm = np.sqrt(vx**2 + vy**2)
            vec_r, v_eta = convert_fpa_to_velocity_components(v_norm, fpa)
            alpha_x, alpha_y = convert_radial_velocity_to_cartesian(
                vec_r, v_eta, self.arr_r_polar_nd[1]
            )
        alpha = np.sqrt(alpha_x**2 + alpha_y**2)
        alpha_x = alpha_x / alpha if alpha != 0 else 0.0
        alpha_y = alpha_y / alpha if alpha != 0 else 0.0
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

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

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="Required step size is less than spacing between numbers",
            )
            solution = solve_ivp(
                env_EOM_TBT_v2, t_span, y0, method="RK45", args=(params,)
            )

        y_final = (solution.y[:, -1]).astype(np.float32)
        delta_r = y_final - y0

        TTG -= self.step_size
        self.elapsed_t += self.step_size
        self._state = np.append(
            y_final,
            [self._state[5], self._state[6], self._state[7], self._state[8], TTG],
        ).astype(np.float32)

        x_plus, y_plus, vx_plus, vy_plus, mass_plus = y_final
        self.mass_increment = mass - mass_plus

        sc.update_state_cartesian(*y_final)
        self._spacecraft = sc

        a, e, w, theta = sc.calc_Planar_OE(*self._arr_cb, mu)
        self._keplerian_elements[:4] = [a, e, w, theta]

        # reward
        if self.reward_mode == "throttle":
            reward, terminated, truncated = self._calc_reward_throttle(u)
        else:
            reward, terminated, truncated = self._calc_reward_mass()

        self.episode_reward += reward

        # observation
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

        if self.obs_mode == "relative":
            observation = self.create_relative_polar_observation(
                x_nd, y_nd, vx_nd, vy_nd, mass_nd, TTG_nd
            )
        else:
            observation = self.create_polar_observation(
                x_nd,
                y_nd,
                vx_nd,
                vy_nd,
                x_target_nd,
                y_target_nd,
                vx_target_nd,
                vy_target_nd,
                mass_nd,
                TTG_nd,
            )

        info = self._get_info(solution, delta_r)
        if terminated or truncated:
            info["episode"] = {"r": float(self.episode_reward)}

        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            raise ValueError("NaN/Inf detected in observation array.")
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError("NaN/Inf detected in reward.")

        return observation, reward, terminated, truncated, info

    def _get_info(self, ode_solution, delta_r):
        """Collect diagnostic info for logging/monitoring."""

        # orbital period of current spacecraft
        a_nd = self._keplerian_elements[0] / Constants.SMA_EARTH
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                T_nd = 2 * np.pi * (a_nd**3) ** 0.5
            except RuntimeWarning:
                T_nd = np.nan

        dx, dy, dr, dvx, dvy, dv, r_target, v_target, dr_norm, dv_norm = (
            self.calc_deltas()
        )

        info = {
            "Elapsed time": self.elapsed_t,
            "ODE Solution": ode_solution,
            "delta_state": delta_r,
            "planet_radii": self.planet_radii,
            "a": self._keplerian_elements[0] / Constants.SMA_EARTH,
            "e": self._keplerian_elements[1],
            "w": np.rad2deg(self._keplerian_elements[2]),
            "theta": np.rad2deg(self._keplerian_elements[3]),
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
            "mu": self.param_mu,
            "step_count": self.step_count,
            "TOF": getattr(self, "TOF", 0.0),
            "TOF_nd": getattr(self, "TOF_nd", 0.0),
            "timesteps_in_TOF": getattr(self, "timesteps_in_TOF", 0),
            "pos_residual": getattr(self, "pos_residual", 0.0),
            "vel_residual": getattr(self, "vel_residual", 0.0),
            "tof_scale": self.tof_scale,
            "r_weight": self.pos_weight,
            "v_weight": self.vel_weight,
            "mass_weight": self.mass_weight,
            "throttle_r_weight": self.throttle_r_weight,
            "pos_reward": getattr(self, "pos_reward", 0.0),
            "vel_reward": getattr(self, "vel_reward", 0.0),
            "mass_reward": getattr(self, "mass_reward", 0.0),
            "throttle_reward": getattr(self, "throttle_reward", 0.0),
            "time_reward": getattr(self, "time_reward", 0.0),
            "alpha_x": getattr(self, "alpha_x", 0.0),
            "alpha_y": getattr(self, "alpha_y", 0.0),
        }

        # include polar snapshot when available (legacy obs mode)
        arr_r = getattr(self, "arr_r_polar_nd", np.zeros(2, dtype=np.float32))
        arr_v = getattr(self, "arr_v_polar_nd", np.zeros(2, dtype=np.float32))
        arr_rf = getattr(self, "arr_rf_polar_nd", np.zeros(2, dtype=np.float32))
        arr_vf = getattr(self, "arr_vf_polar_nd", np.zeros(2, dtype=np.float32))
        info.update(
            {
                "r_nd": arr_r[0] if len(arr_r) > 0 else 0.0,
                "eta_nd": arr_r[1] if len(arr_r) > 1 else 0.0,
                "v_r_nd": arr_v[0] if len(arr_v) > 0 else 0.0,
                "v_eta_nd": arr_v[1] if len(arr_v) > 1 else 0.0,
                "r_nd_f": arr_rf[0] if len(arr_rf) > 0 else 0.0,
                "eta_nd_f": arr_rf[1] if len(arr_rf) > 1 else 0.0,
                "v_r_nd_f": arr_vf[0] if len(arr_vf) > 0 else 0.0,
                "v_eta_nd_f": arr_vf[1] if len(arr_vf) > 1 else 0.0,
            }
        )

        return info

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
        return np.array(
            [
                x_nd,
                y_nd,
                vx_nd,
                vy_nd,
                mass_nd,
                x_target_nd,
                y_target_nd,
                vx_target_nd,
                vy_target_nd,
                TTG_nd,
            ],
            dtype=np.float32,
        )

    def set_state(self, state):
        self._state = np.array(state, dtype=np.float32)
        self._spacecraft.update_state_cartesian(*self._state[0:5])
        a, e, w, theta = self._spacecraft.calc_Planar_OE(*self._arr_cb, self.param_mu)
        self._keplerian_elements[:4] = [a, e, w, theta]
        self.mass_increment = 0.0
        TTG_nd = self._state[9] / self.t_star
        x_nd = self._state[0] / self.l_star
        y_nd = self._state[1] / self.l_star
        vx_nd = self._state[2] / (self.l_star / self.t_star)
        vy_nd = self._state[3] / (self.l_star / self.t_star)
        x_target_nd = self._state[5] / self.l_star
        y_target_nd = self._state[6] / self.l_star
        vx_target_nd = self._state[7] / (self.l_star / self.t_star)
        vy_target_nd = self._state[8] / (self.l_star / self.t_star)
        mass_nd = self._state[4] / self.m_star
        if self.obs_mode == "relative":
            obs = self.create_relative_polar_observation(
                x_nd, y_nd, vx_nd, vy_nd, mass_nd, TTG_nd
            )
        else:
            obs = self.create_polar_observation(
                x_nd,
                y_nd,
                vx_nd,
                vy_nd,
                x_target_nd,
                y_target_nd,
                vx_target_nd,
                vy_target_nd,
                mass_nd,
                TTG_nd,
            )
        info = self._get_info(None, None)
        return obs, info

    def get_cartesian_state(self):
        return self._state.copy()

    def get_target_cartesian_state_at_ttg(self, ttg):
        if self.target_strategy == "static":
            return np.array(
                [
                    self._state[5],
                    self._state[6],
                    self._state[7],
                    self._state[8],
                    1000.0,
                ],
                dtype=np.float32,
            )

        x_target, y_target, vx_target, vy_target = self._state[5:9]
        mass_placeholder = 1000.0
        t_span = (0.0, -ttg)
        y0 = np.array([x_target, y_target, vx_target, vy_target, mass_placeholder])
        params = np.array(
            [
                self.param_mu,
                self._spacecraft.max_thrust,
                self._spacecraft.specific_impulse,
                Constants.G0,
                0.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="Required step size is less than spacing between numbers",
            )
            solution = solve_ivp(
                env_EOM_TBT_v2, t_span, y0, method="RK45", args=(params,)
            )
        y_final = (solution.y[:, -1]).astype(np.float32)
        self.target_state_current = y_final.copy()
        return y_final.copy()


# Temporary aliases for migration convenience
TwoBodyRendezvous_Polar_Env = TwoBodyRendezvousEnv
TwoBodyRendezvous_Polar_Env2 = TwoBodyRendezvousEnv
