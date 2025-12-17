import torch.nn as nn
from stable_baselines3 import SAC as SB3_SAC

from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.utils.path_utils import RUNS_ROOT


def get_model(params, env, seed_in, log_dir, model_id=None):
    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    # load model if specified, otherwise create new
    if model_id is not None:
        print(f"Loading SAC model from: {RUNS_ROOT}/{model_id}")
        model = SB3_SAC.load(
            params["path_SAC_model_load"],
            env=env,
            device="cpu",
            seed=seed_in,
            tensorboard_log=log_dir,
        )  # Use log_dir so SB3 creates SAC_1/ subdirectory
    else:
        # Implement custom NN architectures
        nn_arch_type = params.get("nn_arch_type", "default")
        if nn_arch_type == "custom":
            # use default architecture
            # define the policy architecture
            policy_kwargs = dict(
                net_arch=[32, 32, 32, 32, 32],  # four hidden layers with 32 units each
                activation_fn=nn.LeakyReLU,  # LeakyReLU activation function
                optimizer_kwargs=dict(eps=1e-5),
            )
            model = SB3_SAC(
                "MlpPolicy",
                env,
                learning_rate=params["learning_rate"],
                verbose=1,
                device="cpu",
                seed=seed_in,
                tensorboard_log=log_dir,  # Use log_dir so SB3 creates SAC_1/ subdirectory
                buffer_size=buffer_size,
                tau=params.get("tau", 0.005),
                train_freq=params.get("train_freq", 1),
                gradient_steps=params.get("gradient_steps", 1),
                policy_kwargs=policy_kwargs,
            )

        else:
            # use default architecture
            policy_kwargs = dict(
                optimizer_kwargs=dict(eps=1e-5)  # More stable Adam optimizer
            )
            model = SB3_SAC(
                "MlpPolicy",
                env,
                learning_rate=params["learning_rate"],
                verbose=1,
                device="cpu",
                seed=seed_in,
                tensorboard_log=log_dir,  # Use model_dir so SB3 creates SAC_1/ subdirectory
                buffer_size=buffer_size,
                tau=params.get("tau", 0.005),
                train_freq=params.get("train_freq", 1),
                gradient_steps=params.get("gradient_steps", 1),
                policy_kwargs=policy_kwargs,
            )
    return model
