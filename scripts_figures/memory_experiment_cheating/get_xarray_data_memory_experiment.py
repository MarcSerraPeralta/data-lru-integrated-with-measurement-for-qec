import pathlib

import numpy as np
import xarray as xr

# Parameters
L1_RATES = np.array([0.975, 1.575, 2.6, 3.625, 4.9])  # given by Yuejie
L1_RATE = L1_RATES[3]

EXP_NAME = "20250718_repcode_d3_exp_lru_v2_cheating"
EXPERIMENTS = {
    "logical_0": {
        "cheating": "20250906-115430_vcF4zy_slru_lstm30x2_eval30_b64_dr0-15_lr0-001_injleak3_cheatTrue",
        "no-cheating": "20250926-082144_uOMuUS_slru_lstm30x2_eval20_b64_dr0-10_lr0-001_injleak3_cheatFalse_logical0",
    },
    "logical_0-1": {
        "cheating": "20250522-112934_slru_lstm30x2_eval30_b64_dr0-10_lr0-001",
        "no-cheating": "20250929-105912_kLrjte_slru_lstm30x2_eval20_b64_dr0-10_lr0-001_injleak3_cheatFalse_logical01",
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
LOG_PROBS = np.zeros((2, 2, 1, len(ROUNDS)))  # bitstring, cheating, leakage, rounds
for BITSTRING, DATA in EXPERIMENTS.items():
    for CHEATING, MODEL_NAME in DATA.items():
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

        i = 1 if "1" in BITSTRING else 0
        j = 1 if "no" in CHEATING else 0
        LOG_PROBS[i, j, 0] = log_prob

ds = xr.Dataset(
    data_vars=dict(log_prob=(("bitstrings", "cheating", "l1", "round"), LOG_PROBS)),
    coords=dict(
        bitstrings=["000-011-101-110", "010-101"],
        cheating=["yes", "no"],
        l1=[L1_RATE],
        round=ROUNDS,
    ),
)
ds.to_netcdf("memory_experiment_results_cheating.nc")
