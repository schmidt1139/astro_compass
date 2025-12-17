import os
import pickle

import torch

from astro_compass.core.ephem_converter import (
    read_ephems,
)
from astro_compass.core.ephemeris_v3 import Ephemeris_v3
from astro_compass.utils.path_utils import DATA_ROOT

print("GPU available: ", torch.cuda.is_available())


def main(text_dir, pickle_dir):
    ephems, filenames = read_ephems(
        text_dir,
        eph_class=Ephemeris_v3,
    )

    # save each ephem as a pickle object
    for eph, filename in zip(ephems, filenames):
        path_out = os.path.join(pickle_dir, filename.replace(".txt", ".pkl"))
        path_out = path_out.replace("test_TBR_ephem_traj_seed_", "")
        path_out = path_out.replace("_tof_1p0_scenario_0", "")
        with open(path_out, "wb") as f:
            pickle.dump(eph, f)


if __name__ == "__main__":
    text_dir = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "txt")
    pickle_dir = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "pickle")
    os.makedirs(pickle_dir, exist_ok=True)
    main(text_dir, pickle_dir)
