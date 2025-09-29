from Log_Utils import log


def log_training_perf(test_log, callback, eval_callback, model, training_steps, flag_verbose):

    # --- add final training performance metrics to the log ---
    total_episodes = len(callback.episode_rewards)
    total_timesteps_done = getattr(model, "num_timesteps", training_steps)

    if total_episodes > 0:
        avg_reward = sum(callback.episode_rewards) / total_episodes
        best_reward = max(callback.episode_rewards)
        last_reward = callback.episode_rewards[-1]
    else:
        avg_reward = float("nan")
        best_reward = float("nan")
        last_reward = float("nan")

    if len(callback.episode_lengths) > 0:
        avg_length = sum(callback.episode_lengths) / len(callback.episode_lengths)
    else:
        avg_length = 0

    best_eval = getattr(eval_callback, "best_mean_reward", None)

    test_log = log("FINAL TRAINING METRICS", test_log, flag_verbose)
    test_log = log("Total episodes: " + str(total_episodes), test_log, flag_verbose)
    test_log = log("Total timesteps (model.num_timesteps): " + str(total_timesteps_done), test_log, flag_verbose)
    test_log = log("Average episode reward: " + str(avg_reward), test_log, flag_verbose)
    test_log = log("Best episode reward: " + str(best_reward), test_log, flag_verbose)
    test_log = log("Last episode reward: " + str(last_reward), test_log, flag_verbose)
    test_log = log("Average episode length: " + str(avg_length), test_log, flag_verbose)

    if best_eval is not None:
        test_log = log("Best eval mean reward: " + str(best_eval), test_log, flag_verbose)

    return test_log


