from typing import Dict, Optional

import torch
from stable_baselines3 import SAC as SB3_SAC


def build_sac(
    env,
    model_cfg: Dict,
    training_cfg: Dict,
    seed: int,
    tensorboard_log: Optional[str] = None,
):
    buffer_size = training_cfg.get("buffer_size", 1_000_000)
    train_freq = training_cfg.get("train_freq", model_cfg.get("train_freq", 1))
    train_freq_unit = training_cfg.get(
        "train_freq_unit", model_cfg.get("train_freq_unit", "step")
    )
    gradient_steps = training_cfg.get(
        "gradient_steps", model_cfg.get("gradient_steps", 1)
    )
    learning_starts = training_cfg.get(
        "learning_starts", model_cfg.get("learning_starts", 50_000)
    )
    policy_kwargs = None

    if model_cfg.get("nn_arch_type", "default") == "custom":
        policy_kwargs = dict(
            net_arch=model_cfg.get("net_arch", [256, 256]),
            activation_fn=torch.nn.LeakyReLU,
        )
    else:
        policy_kwargs = dict(
            optimizer_kwargs=dict(eps=1e-5),
        )

    model = SB3_SAC(
        "MlpPolicy",
        env,
        learning_rate=model_cfg["learning_rate"],
        verbose=1,
        device=model_cfg.get("eval_device", "cpu"),
        seed=seed,
        tensorboard_log=tensorboard_log,
        buffer_size=buffer_size,
        tau=model_cfg.get("tau", 0.005),
        train_freq=(train_freq, train_freq_unit),
        gradient_steps=gradient_steps,
        learning_starts=learning_starts,
        policy_kwargs=policy_kwargs,
    )
    return model


__all__ = ["build_sac"]
