from astro_compass.envs.TwoBody_Orb2Orb_Transfer_Env_nd_obs5 import (
    TwoBody_Orb2Orb_Transfer_Env_nd_obs5,
)
from astro_compass.envs.TwoBody_Orb2Orb_Transfer_Env_target import (
    TwoBody_Orb2Orb_Transfer_Env_target,
)
from astro_compass.envs.TwoBodyRendezvous_Polar_Env import TwoBodyRendezvous_Polar_Env
from astro_compass.envs.TwoBodyRendezvous_Polar_Env2 import TwoBodyRendezvous_Polar_Env2


def gen_rl_environment(params):
    if params["env_type"] == "TwoBodyRendezvous_Polar_Env":
        # initialize the environment
        env = TwoBodyRendezvous_Polar_Env(
            mu=params["mu"],
            max_T=params["max_T"],
            ISP=params["ISP"],
            l_star=params["l_star"],
            m_star=params["m_star"],
            t_star=params["t_star"],
            g0=params["g0"],
            step_size=params["env_step_size"],
            a_min_init_env_nd=params["a_min_init_env_nd"],
            a_max_init_env_nd=params["a_max_init_env_nd"],
            e_min_init_env=params["e_min_init_env"],
            e_max_init_env=params["e_max_init_env"],
            w_min_init_env_deg=params["w_min_init_env_deg"],
            w_max_init_env_deg=params["w_max_init_env_deg"],
            a_min_final_env_nd=params["a_min_final_env_nd"],
            a_max_final_env_nd=params["a_max_final_env_nd"],
            e_min_final_env=params["e_min_final_env"],
            e_max_final_env=params["e_max_final_env"],
            w_min_final_env_deg=params["w_min_final_env_deg"],
            w_max_final_env_deg=params["w_max_final_env_deg"],
            pos_r_weight=params.get("pos_r_weight", 1.0),
            vel_r_weight=params.get("vel_r_weight", 1.0),
            mass_r_weight=params.get("mass_r_weight", 1.0),
            tof_scale=params.get("tof_scale", 1.0),
            r_dist_weight=params.get("r_dist_weight", 1.0),
            v_dist_weight=params.get("v_dist_weight", 1.0),
            success_threshold_pos=params.get("success_threshold_pos", 0.01),
            success_threshold_vel=params.get("success_threshold_vel", 0.01),
            terminal_bonus=params.get("terminal_bonus", 100.0),
            precision_mult=params.get("precision_mult", 10.0),
            tof_weight=params.get("tof_weight", 1.0),
            time_dist_weight=params.get("time_dist_weight", 1.0),
            theta_min_init_env_deg=params.get("theta_min_init_env_deg", 0.0),
            theta_max_init_env_deg=params.get("theta_max_init_env_deg", 360.0),
            theta_min_final_env_deg=params.get("theta_min_final_env_deg", 0.0),
            theta_max_final_env_deg=params.get("theta_max_final_env_deg", 360.0),
        )
    elif params["env_type"] == "TwoBodyRendezvous_Polar_Env2":
        # initialize the environment
        env = TwoBodyRendezvous_Polar_Env2(
            mu=params["mu"],
            max_T=params["max_T"],
            ISP=params["ISP"],
            l_star=params["l_star"],
            m_star=params["m_star"],
            t_star=params["t_star"],
            g0=params["g0"],
            step_size=params["env_step_size"],
            a_min_init_env_nd=params["a_min_init_env_nd"],
            a_max_init_env_nd=params["a_max_init_env_nd"],
            e_min_init_env=params["e_min_init_env"],
            e_max_init_env=params["e_max_init_env"],
            w_min_init_env_deg=params["w_min_init_env_deg"],
            w_max_init_env_deg=params["w_max_init_env_deg"],
            a_min_final_env_nd=params["a_min_final_env_nd"],
            a_max_final_env_nd=params["a_max_final_env_nd"],
            e_min_final_env=params["e_min_final_env"],
            e_max_final_env=params["e_max_final_env"],
            w_min_final_env_deg=params["w_min_final_env_deg"],
            w_max_final_env_deg=params["w_max_final_env_deg"],
            pos_r_weight=params.get("pos_r_weight", 1.0),
            vel_r_weight=params.get("vel_r_weight", 1.0),
            throttle_r_weight=params.get("throttle_r_weight", 1.0),
            tof_scale=params.get("tof_scale", 1.0),
            r_dist_weight=params.get("r_dist_weight", 1.0),
            v_dist_weight=params.get("v_dist_weight", 1.0),
            success_threshold_pos=params.get("success_threshold_pos", 0.01),
            success_threshold_vel=params.get("success_threshold_vel", 0.01),
            terminal_bonus=params.get("terminal_bonus", 100.0),
            precision_mult=params.get("precision_mult", 10.0),
            tof_weight=params.get("tof_weight", 1.0),
            time_dist_weight=params.get("time_dist_weight", 1.0),
            theta_min_init_env_deg=params.get("theta_min_init_env_deg", 0.0),
            theta_max_init_env_deg=params.get("theta_max_init_env_deg", 360.0),
            theta_min_final_env_deg=params.get("theta_min_final_env_deg", 0.0),
            theta_max_final_env_deg=params.get("theta_max_final_env_deg", 360.0),
            prop_length_scale=params.get("prop_length_scale", 1.0),
        )
    elif params["env_type"] == "TwoBody_Orb2Orb_Transfer_Env_nd_obs5":
        env = TwoBody_Orb2Orb_Transfer_Env_nd_obs5(**params)

    elif params["env_type"] == "TwoBody_Orb2Orb_Transfer_Env_target":
        # initialize the transfer
        env = TwoBody_Orb2Orb_Transfer_Env_target(**params)
    else:
        raise NotImplementedError(
            f"Environment type {params['env_type']} not implemented."
        )

    return env
