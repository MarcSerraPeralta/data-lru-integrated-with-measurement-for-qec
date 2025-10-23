import pathlib
import os

from qrennd import Config

from lib.evaluation import get_ps_leakage_stability

# Parameters
EXP_NAMES = ["20250619_stability7_exp_lru_v1", "20250619_stability7_exp_no_lru_v1"]
TEST_DATASET = ["test"]

OVERWRITE = False

PARENT_DIR = pathlib.Path.cwd()
DATA_DIR = PARENT_DIR / "data"
OUTPUT_DIR = PARENT_DIR / "output"

####################################


if not DATA_DIR.exists():
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")
if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

for exp_name in EXP_NAMES:
    model_names = os.listdir(OUTPUT_DIR / exp_name)
    model_names = [n for n in model_names if n not in ["__old", ".DS_Store"]]
    model_names = sorted(model_names)

    for model_name in model_names:
        MODEL_DIR = OUTPUT_DIR / exp_name / model_name
        if not MODEL_DIR.exists():
            raise ValueError(f"Model directory does not exist: {MODEL_DIR}")

        CONFIG_FILE = MODEL_DIR / "config.yaml"
        if not CONFIG_FILE.exists():
            raise ValueError(f"Config file does not exist: {CONFIG_FILE}")

        # evaluate the NN
        config = Config.from_yaml(
            filepath=CONFIG_FILE,
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
        )
        if not isinstance(TEST_DATASET, list):
            TEST_DATASET = [TEST_DATASET]

        # if results have not been stored, evaluate model
        for test_dataset in TEST_DATASET:

            NAME_LEAK = f"{test_dataset}_PS-leakage.nc"
            if (MODEL_DIR / NAME_LEAK).exists() and (not OVERWRITE):
                continue

            ps_leak = get_ps_leakage_stability(config, test_dataset)
            ps_leak.to_netcdf(path=MODEL_DIR / NAME_LEAK)
