import os
from datetime import datetime

import astro_compass

PROJECT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(astro_compass.__file__)), ".."
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CONFIG_ROOT = os.path.join(PROJECT_ROOT, "config")
LOGS_ROOT = os.path.join(PROJECT_ROOT, "logs")
PLOT_ROOT = os.path.join(DATA_ROOT, "plots")
RUNS_ROOT = os.path.join(DATA_ROOT, "runs")

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CONFIG_ROOT, exist_ok=True)
os.makedirs(PLOT_ROOT, exist_ok=True)
os.makedirs(RUNS_ROOT, exist_ok=True)
os.makedirs(LOGS_ROOT, exist_ok=True)


def get_run_paths(output_dir):
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250928_143005"
    path_output = os.path.join(output_dir, time_tag)
    path_SAC_model = os.path.join(path_output, "model")
    path_checkpoints = os.path.join(path_output, "checkpoints")
    path_ephems = os.path.join(path_output, "ephems")
    path_plots = os.path.join(path_output, "plots")
    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_ephems, exist_ok=True)
    os.makedirs(path_plots, exist_ok=True)

    return {
        "path_output": path_output,
        "path_SAC_model": path_SAC_model,
        "path_checkpoints": path_checkpoints,
        "path_ephems": path_ephems,
        "path_plots": path_plots,
        "id": time_tag,
    }
