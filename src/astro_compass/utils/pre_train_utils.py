import os

import numpy as np
import pandas as pd
from stable_baselines3.common.logger import Logger
from tqdm import tqdm


def get_losses(model, critic_losses, actor_losses):
    try:
        # Check if _logger attribute exists (not the property)
        if hasattr(model, "_logger") and model._logger is not None:
            # Try to get values from logger's name_to_value dict
            if hasattr(model._logger, "name_to_value"):
                critic_loss = model._logger.name_to_value.get("train/critic_loss", None)
                actor_loss = model._logger.name_to_value.get("train/actor_loss", None)

                if critic_loss is not None:
                    critic_losses.append(critic_loss)
                if actor_loss is not None:
                    actor_losses.append(actor_loss)
    except (AttributeError, KeyError):
        # Logger not set up or accessible, skip logging
        pass
    return critic_losses, actor_losses


def report_losses(step, log_interval, critic_losses, actor_losses):
    if (step) % log_interval == 0:
        if critic_losses and actor_losses:
            avg_critic = np.mean(critic_losses[-log_interval:])
            avg_actor = np.mean(actor_losses[-log_interval:])
            tqdm.write(
                f"Step {step + 1}: Critic Loss: {avg_critic:.4f}, Actor Loss: {avg_actor:.4f}"
            )


def train_on_replay_buffer(model, params, env, paths):
    num_gradient_steps = params.get("pretrain_gradient_steps", 10000)
    batch_size = params.get("batch_size", 256)

    print(f"Buffer Transitions: {model.replay_buffer.size()}")
    print(f"Buffer Size: {model.replay_buffer.size}")

    print(
        f"Training networks on replay buffer with {num_gradient_steps} gradient steps",
    )

    # Check that we have enough samples
    buffer_size = model.replay_buffer.size()
    if buffer_size < batch_size:
        print(f"Replay buffer size: {buffer_size}, Batch size: {batch_size}")
        # Return empty loss arrays if we can't train
        return [], []

    print(f"Replay buffer size: {buffer_size}")
    print(f"Batch size: {batch_size}")

    model._logger = Logger(folder=None, output_formats=[])

    # Set up progress tracking (needed by some internal SB3 methods)
    model._current_progress_remaining = 1.0

    # Set the number of timesteps so model.train() knows where we are
    model._total_timesteps = 0
    model.num_timesteps = 0

    # Track losses
    critic_losses = []
    actor_losses = []
    mean_rewards = []
    std_rewards = []

    # Optionally log progress
    log_interval = params.get("pt_log_interval", 1000)
    checkpoint_interval = params.get("checkpoint_freq", 10_000)
    gradient_batch_size = params.get("batch_size", 100)

    # Train the networks by sampling from replay buffer
    for step in tqdm(range(0, num_gradient_steps), desc="Network updates"):
        # This performs multiple gradient updates on actor and critic networks at once
        steps_to_train = min(gradient_batch_size, num_gradient_steps - step)

        model.train(gradient_steps=steps_to_train, batch_size=batch_size)

        # Try to get losses from the model's internal tracking
        # Note: SAC stores these internally but they may not be accessible immediately
        critic_losses, actor_losses = get_losses(model, critic_losses, actor_losses)
        report_losses(step, log_interval, critic_losses, actor_losses)

        # also checkpoint the model
        if (step) % checkpoint_interval == 0:
            # evaluate current policy
            mean_reward, std_reward = evaluate_policy(
                model,
                env,
                n_eval_episodes=params.get("n_eval_episodes", 10),
                deterministic=True,
            )

            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)

            tqdm.write(
                f"Evaluation after {step} steps: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}"
            )

            if mean_reward >= max(mean_rewards, default=-np.inf):
                checkpoint_path = os.path.join(
                    paths["path_checkpoint"], f"sac_pretrained_step_{step + 1}.zip"
                )
                model.save(checkpoint_path)
                tqdm.write(f"Saved model checkpoint to {checkpoint_path}")

    # Save losses (downsample by 10x for file size)
    critic_loss_reduced = []
    actor_loss_reduced = []
    for i in range(len(critic_losses)):
        if i % 10 == 0:
            critic_loss_reduced.append(critic_losses[i])
            actor_loss_reduced.append(actor_losses[i])

    # make a single dataframe with all of the losses + rewards
    df = pd.DataFrame(
        {
            "critic_loss": critic_loss_reduced,
            "actor_loss": actor_loss_reduced,
        }
    )
    df.to_csv(os.path.join(checkpoint_dir, "pretraining_losses.csv"), index=False)

    df = pd.DataFrame(
        {
            "mean_reward": mean_rewards,
            "std_reward": std_rewards,
        }
    )
    df.to_csv(os.path.join(checkpoint_dir, "eval_rewards.csv"), index=True)
    return model
