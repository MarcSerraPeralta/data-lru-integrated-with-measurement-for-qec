import pathlib

import numpy as np
import xarray as xr

# Parameters
L1_RATES = np.array([0.975, 1.575, 2.6, 3.625, 4.9])  # given by Yuejie
EXPERIMENTS = {
    "20250718_repcode_d3_exp_lru_v2": {
        "SLRU": [
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        ],
        "NSLRU": [
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        ],
    },
    "20250718_repcode_d3_exp_no_lru_v2": {
        "SLRU": [
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
            "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        ],
        "NSLRU": [
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
            "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        ],
    },
}
TEST_DATASET = "test"

PARENT_DIR = pathlib.Path.cwd()
DATA_DIR = PARENT_DIR / "data"
OUTPUT_DIR = PARENT_DIR / "output"

####################################

if not DATA_DIR.exists():
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")
if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

ROUNDS = np.array(
    [1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 21, 26, 33, 42, 53, 68, 87, 108], dtype=int
)
LOG_PROBS = np.zeros((2, 2, 5, len(ROUNDS)))  # lru, signaling, leakage, rounds
for EXP_NAME, DATA in EXPERIMENTS.items():
    for NAME in DATA:
        for k, MODEL_NAME in enumerate(DATA[NAME]):
            MODEL_DIR = OUTPUT_DIR / EXP_NAME / MODEL_NAME
            if not MODEL_DIR.exists():
                raise ValueError(f"Model directory does not exist: {MODEL_DIR}")

            NAME_NC = f"{TEST_DATASET}.nc"
            if not (MODEL_DIR / NAME_NC).exists():
                raise ValueError(f"No dataset found for {MODEL_DIR / NAME_NC}")

            log_fid = xr.load_dataset(MODEL_DIR / NAME_NC)

            log_prob = log_fid.log_errors.transpose("qec_round", "state", "shot").values
            log_prob = log_prob.mean(axis=(1, 2))
            rounds = log_fid.qec_round.values
            assert (ROUNDS == rounds).all()

            i = 1 if "no_lru" in EXP_NAME else 0
            j = 1 if "NSLRU" in NAME else 0
            k = k
            LOG_PROBS[i, j, k] = log_prob

ds = xr.Dataset(
    data_vars=dict(log_prob=(("lru", "ro", "l1", "round"), LOG_PROBS)),
    coords=dict(lru=["yes", "no"], ro=["3ro", "2ro"], l1=L1_RATES, round=ROUNDS),
)
ds.to_netcdf("memory_experiment_results.nc")
