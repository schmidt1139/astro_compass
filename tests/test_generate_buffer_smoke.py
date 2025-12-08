import os
import runpy
import shutil
import tempfile
from pathlib import Path

from utils.config_utils import load_config
from utils.path_utils import PROJECT_ROOT, ensure_repo_paths_on_sys_path


def test_generate_buffer_smoke():
    """Run generate_buffer with tiny settings to ensure it executes."""
    ensure_repo_paths_on_sys_path()

    prev_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        base_files = ["common.toml", "envs.toml", "models.toml", "pretraining.toml"]
        experiment_file = "experiments/generate_buffer_default.toml"
        params, meta = load_config(base_files, experiment_file)

        # Fast overrides
        params.update(
            {
                "training_steps": 50,
                "num_ephems_to_use": 1,
                "buffer_size": 500,
                "batch_size": 64,
                "num_vec_envs": 1,
                "cores": 1,
                "path_training_data": "data/pre-training-data/training-TBR-overfit",  # small test set
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
        main_fn(params, seed_in=0, config_meta=meta)

    finally:
        os.chdir(prev_cwd)
        if "tmp_output" in locals() and tmp_output.exists():
            shutil.rmtree(tmp_output, ignore_errors=True)
