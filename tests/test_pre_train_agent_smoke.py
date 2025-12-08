import os
import runpy
import shutil
import tempfile
from pathlib import Path

from utils.config_utils import load_config
from utils.env_utils import gen_rl_environment
from utils.path_utils import PROJECT_ROOT, ensure_repo_paths_on_sys_path


def _make_dummy_replay_buffer(params):
    env = gen_rl_environment(params)
    model_mod = runpy.run_path(
        str(PROJECT_ROOT / "src" / "scripts" / "pretrain" / "train_agent.py")
    )
    SB3_SAC = model_mod["SB3_SAC"]
    model = SB3_SAC(
        "MlpPolicy",
        env,
        verbose=0,
        buffer_size=100,
        device=params.get("eval_device", "cpu"),
    )
    # collect one transition to make buffer non-empty
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs2, reward, terminated, truncated, info = env.step(action)
    # ReplayBuffer expects a list of info dicts (one per env); wrap the single info.
    model.replay_buffer.add(
        obs,
        obs2,
        action,
        reward,
        terminated or truncated,
        [info],
    )
    tmp_rb = (
        Path(tempfile.mkdtemp(prefix="rb_tmp_", dir=PROJECT_ROOT / "data" / "output"))
        / "replay_buffer.pkl"
    )
    model.save_replay_buffer(tmp_rb)
    env.close()
    return tmp_rb


def test_pre_train_agent_smoke():
    """Run pre_train_agent with tiny settings to ensure it executes."""
    ensure_repo_paths_on_sys_path()

    prev_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        base_files = ["common.toml", "envs.toml", "models.toml", "pretraining.toml"]
        experiment_file = "experiments/pretrain_default.toml"
        params, meta = load_config(base_files, experiment_file)

        params.update(
            {
                "training_steps": 50,
                "pretrain_gradient_steps": 50,
                "buffer_size": 500,
                "batch_size": 64,
                "num_vec_envs": 1,
                "cores": 1,
                "max_episode_steps": 64,
                "eval_device": "cpu",
                "config_toml": "pre_train_config.toml",
                "path_replay_buffer": "",  # will be replaced below
            }
        )

        tmp_output = Path(
            tempfile.mkdtemp(
                prefix="pretrain_smoke_", dir=PROJECT_ROOT / "data" / "output"
            )
        )
        params["output_dir"] = str(tmp_output)

        # Prepare a tiny replay buffer to satisfy load_replay_buffer
        tmp_rb = _make_dummy_replay_buffer(params)
        params["path_replay_buffer"] = str(tmp_rb)

        mod = runpy.run_path(
            str(PROJECT_ROOT / "src" / "scripts" / "pretrain" / "pre_train_agent.py")
        )
        main_fn = mod["main"]

        main_fn(params, seed_in=0, config_meta=meta)

    finally:
        os.chdir(prev_cwd)
        if "tmp_output" in locals() and tmp_output.exists():
            shutil.rmtree(tmp_output, ignore_errors=True)
        if "tmp_rb" in locals() and Path(tmp_rb).exists():
            Path(tmp_rb).unlink(missing_ok=True)
