import os
from datetime import datetime

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.env_utils import gen_rl_environment
from utils.path_utils import PROJECT_ROOT


def generate_env(config, seed_in):
    env_cfg = config["environment"]
    episode_cfg = env_cfg.get("episode", {})
    max_steps = episode_cfg.get("max_episode_steps", 5000)
    vec_cfg = env_cfg.get("vectorization", {})
    num_envs = vec_cfg.get("num_vec_envs", 1)

    def single_env_make_env(seed):
        env = gen_rl_environment(config)
        env.seed(seed)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env = Monitor(env)
        return env

    def make_env(seed):
        def _init():
            return single_env_make_env(seed)

        return _init

    single_env = single_env_make_env(seed_in)
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # establish eval environment
    eval_env = gen_rl_environment(config)
    pre_train_env = gen_rl_environment(config)
    pre_train_env.seed(seed_in)
    pre_train_env.reset()
    pre_train_env = gym.wrappers.TimeLimit(pre_train_env, max_episode_steps=max_steps)
    pre_train_env = Monitor(pre_train_env)

    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_steps)
    eval_env = Monitor(eval_env)

    return env, eval_env, pre_train_env, single_env


def generate_paths(paths_cfg):
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250928_143005"
    path_nns = os.path.normpath(os.path.join(PROJECT_ROOT, "data", "neural_networks"))

    # Handle both absolute and relative paths for output_dir
    output_base = paths_cfg["output_dir"]
    if not os.path.isabs(output_base):
        output_base = os.path.join(PROJECT_ROOT, output_base)
    path_output = os.path.normpath(
        os.path.join(output_base, "SAC_training_TBR_polar" + time_tag),
    )

    path_SAC_model = os.path.normpath(os.path.join(path_nns, "sac_tbr_polar_model"))
    os.makedirs(path_output, exist_ok=True)
    paths_cfg["output_dir_specific"] = path_output

    # make a subdir for checkpoints
    path_checkpoints = os.path.normpath(os.path.join(path_output, "checkpoints"))
    path_ephems = os.path.normpath(os.path.join(path_output, "ephems"))
    path_plots = os.path.normpath(os.path.join(path_output, "plots"))
    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_ephems, exist_ok=True)
    os.makedirs(path_plots, exist_ok=True)

    return path_output, path_SAC_model, path_plots
