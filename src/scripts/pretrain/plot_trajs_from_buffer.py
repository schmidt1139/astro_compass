import torch
from utils.buffer_utils import extract_episodes_from_buffer, plot_episodes
from utils.config_utils import load_config
from utils.env_utils import gen_rl_environment
from utils.model_utils import build_sac
from utils.pretrain_utils import generate_paths
from utils.rl_utils import _flatten_config_params

print("GPU available: ", torch.cuda.is_available())


def main(config, seed_in=42, config_meta=None):
    print("Plot Trajectories from Replay Buffer Script")
    print("")

    paths_cfg = config["paths"]
    model_cfg = config["model"]
    training_cfg = config.get("training", {})

    env = gen_rl_environment(config)

    path_output, path_SAC_model, path_plots = generate_paths(paths_cfg)
    paths_cfg["path_plots"] = path_plots

    model = build_sac(
        env=env,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        seed=seed_in,
        tensorboard_log=path_output,
    )

    print("Loading the replay buffer: " + paths_cfg["path_replay_buffer"])
    model.load_replay_buffer(paths_cfg["path_replay_buffer"])
    print("Replay buffer loaded.")
    transitions = model.replay_buffer.size()
    print(f"Number of transitions in replay buffer: {transitions}")

    print("Extracting episodes from the replay buffer...")
    flat_params = _flatten_config_params(config)
    episodes = extract_episodes_from_buffer(model.replay_buffer, flat_params)
    print(f"Extracted {len(episodes)} episodes from the replay buffer.")

    plot_episodes(episodes, flat_params, path_plots)
    print("Plots saved to: " + path_plots)


if __name__ == "__main__":
    base_files = [
        "common.toml",
        "envs.toml",
        "models.toml",
        "pretraining.toml",
    ]
    experiment_file = "experiments/generate_buffer_default.toml"
    config, meta = load_config(base_files=base_files, experiment_file=experiment_file)

    main(config, seed_in=0, config_meta=meta)
