import os
import runpy
import shutil
import tempfile
from pathlib import Path

import torch
from utils.log_utils import read_toml_config_file
from utils.path_utils import PROJECT_ROOT, ensure_repo_paths_on_sys_path


def test_generate_buffer_smoke():
    """Run generate_buffer with tiny settings to ensure it executes."""
    ensure_repo_paths_on_sys_path()

    prev_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        config_path = PROJECT_ROOT / "data" / "config" / "gen_buffer_config.toml"
        params = read_toml_config_file(str(config_path))

        # Fast overrides
        params.update(
            {
                "training_steps": 50,
                "num_ephems_to_use": 1,
                "buffer_size": 500,
                "batch_size": 64,
                "num_vec_envs": 1,
                "cores": 1,
                "path_training_data": "data/test_data/test_datagen_Hamiltonian_TBR_parallel",  # small test set
                "eval_device": "cpu",
                "config_toml": "gen_buffer_config.toml",
            }
        )

        tmp_output = Path(
            tempfile.mkdtemp(
                prefix="gen_buffer_smoke_", dir=PROJECT_ROOT / "data" / "output"
            )
        )
        params["output_dir"] = str(tmp_output)

        mod = runpy.run_path(
            str(PROJECT_ROOT / "src" / "scripts" / "pretrain" / "generate_buffer.py")
        )
        main_fn = mod["main"]

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        main_fn(params, params["path_training_data"], seed_in=0)

        replay_path = tmp_output / "replay_buffer.pkl"
        assert replay_path.exists(), "Replay buffer not saved"
    finally:
        os.chdir(prev_cwd)
        if "tmp_output" in locals() and tmp_output.exists():
            shutil.rmtree(tmp_output, ignore_errors=True)
