import os
import runpy
import shutil
import tempfile
from pathlib import Path

from utils.config_utils import load_config
from utils.path_utils import PROJECT_ROOT, ensure_repo_paths_on_sys_path


def test_train_agent_smoke():
    """Run a tiny training loop to ensure train_agent executes end-to-end."""
    ensure_repo_paths_on_sys_path()

    # Work from repo root so relative paths in generate_paths() resolve correctly.
    prev_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        base_files = ["common.toml", "envs.toml", "models.toml", "training.toml"]
        experiment_file = "experiments/train_default.toml"
        config, meta = load_config(base_files, experiment_file)

        training = config["training"]
        env_cfg = config["environment"]
        paths_cfg = config["paths"]
        general_cfg = config.get("general", {})
        vec_cfg = env_cfg.setdefault("vectorization", {})
        episode_cfg = env_cfg.setdefault("episode", {})

        # Minimal compute footprint
        training.update(
            {
                "training_steps": 100,
                "eval_freq": 50,
                "n_eval_episodes": 1,
                "buffer_size": 1000,
                "batch_size": 32,
                "read_replay_buffer": False,
                "load_model_checkpoint": False,
                "save_final_replay_buffer": False,
                "flag_checkpoint_replay_buffer": False,
            }
        )

        vec_cfg["num_vec_envs"] = 1
        general_cfg["cores"] = 1
        episode_cfg["max_episode_steps"] = 64
        env_cfg["env_step_size"] = env_cfg.get("env_step_size", 1209600)

        # Route outputs to a temporary directory inside the repo for cleanup.
        tmp_output = Path(
            tempfile.mkdtemp(
                prefix="train_agent_smoke_", dir=PROJECT_ROOT / "data" / "output"
            )
        )
        paths_cfg["output_dir"] = str(tmp_output)
        config["config_toml"] = "train_agent_config.toml"

        # Import train_agent via runpy to avoid package path issues.
        mod = runpy.run_path(
            str(PROJECT_ROOT / "src" / "scripts" / "pretrain" / "train_agent.py")
        )
        train_main = mod["main"]

        train_main(config, seed_in=0, config_meta=meta)

        # Verify outputs were written
        assert tmp_output.exists(), "Output directory was not created"

    finally:
        os.chdir(prev_cwd)
        # Clean up temp output
        if "tmp_output" in locals() and tmp_output.exists():
            shutil.rmtree(tmp_output, ignore_errors=True)


if __name__ == "__main__":
    test_train_agent_smoke()
