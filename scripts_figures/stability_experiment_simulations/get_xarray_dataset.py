import pathlib

import numpy as np
import xarray as xr

# Parameters
SWEEP = "resf_sweep"
L1_RATES = np.array([0, 1, 2, 3, 4, 5])  # given by Yuejie

EXPERIMENTS = {
    f"20251022_stability7_simulation_{SWEEP}_lru_v2": {
        "SLRU": [
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak5",
        ],
        "NSLRU": [
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak5",
        ],
    },
    f"20251022_stability7_simulation_{SWEEP}_no_lru_v2": {
        "SLRU": [
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
            "20251022_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak5",
        ],
        "NSLRU": [
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
            "20251022_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak5",
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
    [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
    ]
)
LOG_PROBS = np.zeros((2, 2, 6, len(ROUNDS)))  # lru, signaling, leakage, round
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
            assert (rounds == ROUNDS).all()

            i = 1 if "no_lru" in EXP_NAME else 0
            j = 1 if "nslru" in MODEL_NAME else 0
            k = k
            LOG_PROBS[i, j, k] = log_prob

ds = xr.Dataset(
    data_vars=dict(log_prob=(("lru", "ro", "l1", "round"), LOG_PROBS)),
    coords=dict(lru=["yes", "no"], ro=["3ro", "2ro"], l1=L1_RATES, round=ROUNDS),
)
ds.to_netcdf(f"stability_experiment_{SWEEP}.nc")
