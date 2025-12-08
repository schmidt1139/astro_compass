import os
import runpy
import shutil
import tempfile
from pathlib import Path

from utils.config_utils import load_config
from utils.env_utils import gen_rl_environment
from utils.path_utils import PROJECT_ROOT, ensure_repo_paths_on_sys_path


def _make_dummy_model(params):
    env = gen_rl_environment(params)
    agent_mod = runpy.run_path(
        str(PROJECT_ROOT / "src" / "scripts" / "pretrain" / "train_agent.py")
    )
    SB3_SAC = agent_mod["SB3_SAC"]
    model = SB3_SAC(
        "MlpPolicy",
        env,
        verbose=0,
        buffer_size=100,
        device=params.get("eval_device", "cpu"),
    )
    tmp_model_dir = Path(
        tempfile.mkdtemp(prefix="eval_model_", dir=PROJECT_ROOT / "data" / "output")
    )
    tmp_model = tmp_model_dir / "best_model.zip"
    model.save(tmp_model)
    env.close()
    return tmp_model, tmp_model_dir


def test_evaluate_agent_smoke():
    """Run evaluate_agent with tiny settings to ensure it executes."""
    ensure_repo_paths_on_sys_path()

    prev_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    tmp_model_dir = None
    try:
        base_files = ["common.toml", "envs.toml", "models.toml", "training.toml"]
        experiment_file = "experiments/eval_default.toml"
        params, meta = load_config(base_files, experiment_file)

        params.update(
            {
                "training_steps": 50,
                "eval_freq": 25,
                "n_eval_episodes": 1,
                "buffer_size": 500,
                "batch_size": 64,
                "num_vec_envs": 1,
                "cores": 1,
                "num_rollouts": 1,
                "max_episode_steps": 64,
                "eval_device": "cpu",
                "config_toml": "evaluate_agent_config.toml",
                "flag_gen_H_traj": False,
            }
        )

        tmp_output = Path(
            tempfile.mkdtemp(prefix="eval_smoke_", dir=PROJECT_ROOT / "data" / "output")
        )
        params["output_dir"] = str(tmp_output)

        tmp_model, tmp_model_dir = _make_dummy_model(params)
        params["path_SAC_model_load"] = str(tmp_model)

        mod = runpy.run_path(
            str(PROJECT_ROOT / "src" / "scripts" / "pretrain" / "evaluate_agent.py")
        )
        main_fn = mod["main"]
        main_fn(params, seed_in=0, config_meta=meta)

        saved_config = tmp_output / params["config_toml"]
        assert saved_config.exists(), "Config copy missing in output"
    finally:
        os.chdir(prev_cwd)
        if "tmp_output" in locals() and tmp_output.exists():
            shutil.rmtree(tmp_output, ignore_errors=True)
        if tmp_model_dir and tmp_model_dir.exists():
            shutil.rmtree(tmp_model_dir, ignore_errors=True)
