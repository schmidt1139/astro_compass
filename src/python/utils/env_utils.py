from envs.TwoBody_Orb2Orb_Transfer_Env_target import TwoBody_Orb2Orb_Transfer_Env_target
from envs.TwoBodyRendezvousEnv import TwoBodyRendezvousEnv


def _env_common_kwargs(env_cfg, reward_cfg):
    return dict(
        mu=env_cfg["mu"],
        max_T=env_cfg["max_T"],
        ISP=env_cfg["ISP"],
        l_star=env_cfg["l_star"],
        m_star=env_cfg["m_star"],
        t_star=env_cfg["t_star"],
        g0=env_cfg.get("g0", 9.80665),
        step_size=env_cfg["env_step_size"],
        pos_r_weight=reward_cfg.get("pos_r_weight", 1.0),
        vel_r_weight=reward_cfg.get("vel_r_weight", 1.0),
        throttle_r_weight=reward_cfg.get(
            "throttle_r_weight", reward_cfg.get("mass_r_weight", 1.0)
        ),
        tof_scale=reward_cfg.get("tof_scale", 1.0),
        r_dist_weight=reward_cfg.get("r_dist_weight", 1.0),
        v_dist_weight=reward_cfg.get("v_dist_weight", 1.0),
        success_threshold_pos=reward_cfg.get("success_threshold_pos", 0.01),
        success_threshold_vel=reward_cfg.get("success_threshold_vel", 0.01),
        terminal_bonus=reward_cfg.get("terminal_bonus", 0.0),
        precision_mult=reward_cfg.get("precision_mult", 1.0),
        tof_weight=reward_cfg.get("tof_weight", 1.0),
        time_dist_weight=reward_cfg.get("time_dist_weight", 0.0),
        prop_length_scale=reward_cfg.get("prop_length_scale", 1.0),
    )


def gen_rl_environment(config):
    env_cfg = config["environment"]
    reward_cfg = env_cfg.get("reward", {})
    init_cfg = env_cfg.get("initial", {})
    final_cfg = env_cfg.get("final", {})

    if env_cfg["env_type"] in (
        "TwoBodyRendezvousEnv",
        "TwoBodyRendezvous_Polar_Env",
        "TwoBodyRendezvous_Polar_Env2",
    ):
        legacy = env_cfg["env_type"] == "TwoBodyRendezvous_Polar_Env"
        env = TwoBodyRendezvousEnv(
            a_min_init_env_nd=init_cfg["a_min_init_env_nd"],
            a_max_init_env_nd=init_cfg["a_max_init_env_nd"],
            e_min_init_env=init_cfg["e_min_init_env"],
            e_max_init_env=init_cfg["e_max_init_env"],
            w_min_init_env_deg=init_cfg["w_min_init_env_deg"],
            w_max_init_env_deg=init_cfg["w_max_init_env_deg"],
            a_min_final_env_nd=final_cfg["a_min_final_env_nd"],
            a_max_final_env_nd=final_cfg["a_max_final_env_nd"],
            e_min_final_env=final_cfg["e_min_final_env"],
            e_max_final_env=final_cfg["e_max_final_env"],
            w_min_final_env_deg=final_cfg["w_min_final_env_deg"],
            w_max_final_env_deg=final_cfg["w_max_final_env_deg"],
            theta_min_init_env_deg=init_cfg.get("theta_min_init_env_deg", 0.0),
            theta_max_init_env_deg=init_cfg.get("theta_max_init_env_deg", 360.0),
            theta_min_final_env_deg=final_cfg.get("theta_min_final_env_deg", 0.0),
            theta_max_final_env_deg=final_cfg.get("theta_max_final_env_deg", 360.0),
            target_strategy="static" if legacy else "propagated",
            action_mode="fpa" if legacy else "radial",
            obs_mode="polar" if legacy else "relative",
            reward_mode="mass" if legacy else "throttle",
            **_env_common_kwargs(env_cfg, reward_cfg),
            flag_hyperbolic_termination=env_cfg.get(
                "flag_hyperbolic_termination", False
            ),
            max_episode_steps=env_cfg.get("max_episode_steps", 5000),
        )
    elif env_cfg["env_type"] == "TwoBody_Orb2Orb_Transfer_Env_target":
        # initialize the transfer
        env = TwoBody_Orb2Orb_Transfer_Env_target(
            a_min_init_env_nd=init_cfg["a_min_init_env_nd"],
            a_max_init_env_nd=init_cfg["a_max_init_env_nd"],
            e_min_init_env=init_cfg["e_min_init_env"],
            e_max_init_env=init_cfg["e_max_init_env"],
            w_min_init_env_deg=init_cfg["w_min_init_env_deg"],
            w_max_init_env_deg=init_cfg["w_max_init_env_deg"],
            a_min_final_env_nd=final_cfg["a_min_final_env_nd"],
            a_max_final_env_nd=final_cfg["a_max_final_env_nd"],
            e_min_final_env=final_cfg["e_min_final_env"],
            e_max_final_env=final_cfg["e_max_final_env"],
            w_min_final_env_deg=final_cfg["w_min_final_env_deg"],
            w_max_final_env_deg=final_cfg["w_max_final_env_deg"],
            theta_min_init_env_deg=init_cfg.get("theta_min_init_env_deg", 0.0),
            theta_max_init_env_deg=init_cfg.get("theta_max_init_env_deg", 360.0),
            **_env_common_kwargs(env_cfg, reward_cfg),
        )
    else:
        raise NotImplementedError(
            f"Environment type {env_cfg['env_type']} not implemented."
        )

    return env
