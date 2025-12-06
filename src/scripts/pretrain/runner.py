import os

import torch
import utils
from evaluate_agent import main as main_evaluate_agent
from pre_train_agent import main as main_pre_train_agent
from train_agent import main as main_train_agent
from utils.log_utils import read_toml_config_file

print("GPU available: ", torch.cuda.is_available())
# HACK
PROJECT_ROOT = os.path.dirname(os.path.dirname(utils.__file__)) + "/../.."


def main():
    config_toml = "train_agent_config.toml"
    path_config = os.path.join(PROJECT_ROOT, "data", "config", config_toml)
    params = read_toml_config_file(path_config)

    environments = ["TBR_esay_fixed", "TBR_medium_fixed", "TBR_hard_fixed"]
    environments += ["TBR_esay_variable", "TBR_medium_variable", "TBR_hard_variable"]
    pre_training_types = ["none", "small", "medium", "large"]

    for env in environments:
        for pre_train in pre_training_types:
            dataset_name = f"{pre_train}_{env}_dataset"
            params["pre_training_dataset"] = dataset_name
            params["pre_training_type"] = pre_train
            params["environment"] = env

    if not pre_train == "none":
        main_pre_train_agent(params, seed_in=42)

    main_train_agent(params, seed_in=42)
    main_evaluate_agent(params, seed_in=42)


if __name__ == "__main__":
    main()
