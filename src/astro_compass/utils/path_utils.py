import os

import astro_compass

PROJECT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(astro_compass.__file__)), ".."
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CONFIG_ROOT = os.path.join(PROJECT_ROOT, "config")
PLOT_ROOT = os.path.join(DATA_ROOT, "plots")
RUNS_ROOT = os.path.join(DATA_ROOT, "runs")

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CONFIG_ROOT, exist_ok=True)
os.makedirs(PLOT_ROOT, exist_ok=True)
os.makedirs(RUNS_ROOT, exist_ok=True)
