import os
import runpy
import shutil
import tempfile
from pathlib import Path

import torch
from utils.log_utils import read_toml_config_file
from utils.path_utils import PROJECT_ROOT, ensure_repo_paths_on_sys_path


def test_train_agent_smoke():
    """Run a tiny training loop to ensure train_agent executes end-to-end."""
    ensure_repo_paths_on_sys_path()

    # Work from repo root so relative paths in generate_paths() resolve correctly.
    prev_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        # Load base config and override for a fast smoke run.
        config_path = PROJECT_ROOT / "data" / "config" / "train_agent_config.toml"
        params = read_toml_config_file(str(config_path))

        # Minimal compute footprint
        params.update(
            {
                "training_steps": 100,
                "eval_freq": 50,
                "n_eval_episodes": 1,
                "buffer_size": 1000,
                "batch_size": 32,
                "num_vec_envs": 1,
                "cores": 1,
                "read_replay_buffer": False,
                "load_model_checkpoint": False,
                "save_final_replay_buffer": False,
                "flag_checkpoint_replay_buffer": False,
                "max_episode_steps": 64,
                # keep env step small to avoid long trajectories
                "env_step_size": params.get("env_step_size", 1209600),
            }
        )

        # Route outputs to a temporary directory inside the repo for cleanup.
        tmp_output = Path(
            tempfile.mkdtemp(
                prefix="train_agent_smoke_", dir=PROJECT_ROOT / "data" / "output"
            )
        )
        params["output_dir"] = str(tmp_output)
        params["config_toml"] = "train_agent_config.toml"

        # Import train_agent via runpy to avoid package path issues.
        mod = runpy.run_path(
            str(PROJECT_ROOT / "src" / "scripts" / "pretrain" / "train_agent.py")
        )
        train_main = mod["main"]

        # Limit torch parallelism for predictability in CI
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        train_main(params, seed_in=0)

        # Verify outputs were written
        assert tmp_output.exists(), "Output directory was not created"
        # model path is set inside train_agent via generate_paths
        model_path = Path(params.get("output_dir_specific", tmp_output)) / "checkpoints"
        saved_config = tmp_output / params["config_toml"]
        assert saved_config.exists(), "Config copy missing in output"
    finally:
        os.chdir(prev_cwd)
        # Clean up temp output
        if "tmp_output" in locals() and tmp_output.exists():
            shutil.rmtree(tmp_output, ignore_errors=True)
