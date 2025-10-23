import pathlib

import numpy as np
import xarray as xr

# Parameters
L1_RATES = np.array([1.2, 2.8, 3.7, 5.3, 7.6])  # given by Yuejie

EXPERIMENTS = {
    "20250619_stability7_exp_lru_v1": {
        "SLRU": [
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        ],
        "NSLRU": [
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        ],
    },
    "20250619_stability7_exp_no_lru_v1": {
        "SLRU": [
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
            "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        ],
        "NSLRU": [
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
            "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
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

ROUNDS = np.arange(3, 41, dtype=int)
LOG_PROBS_LEAKAGE = np.zeros((2, 2, 5, len(ROUNDS)))  # lru, signaling, leakage, rounds
LOG_PROBS_CONFIDENCE = np.zeros(
    (2, 2, 5, len(ROUNDS))
)  # lru, signaling, leakage, rounds
NUM_SHOTS = np.zeros(
    (2, 2, 5, len(ROUNDS)), dtype=int
)  # lru, signaling, leakage, rounds
for EXP_NAME, DATA in EXPERIMENTS.items():
    for NAME in DATA:
        for k, MODEL_NAME in enumerate(DATA[NAME]):
            MODEL_DIR = OUTPUT_DIR / EXP_NAME / MODEL_NAME
            if not MODEL_DIR.exists():
                raise ValueError(f"Model directory does not exist: {MODEL_DIR}")

            NAME_NC = f"{TEST_DATASET}.nc"
            NAME_LEAK = f"{TEST_DATASET}_PS-leakage.nc"
            if not (MODEL_DIR / NAME_NC).exists():
                raise ValueError(f"No dataset found for {MODEL_DIR / NAME_NC}")
            if not (MODEL_DIR / NAME_LEAK).exists():
                raise ValueError(f"No dataset found for {MODEL_DIR / NAME_LEAK}")

            log_fid = xr.load_dataset(MODEL_DIR / NAME_NC)
            ps_leak = xr.load_dataset(MODEL_DIR / NAME_LEAK)
            log_errors = log_fid.log_errors.transpose(
                "qec_round", "state", "shot"
            ).values.astype(int)
            soft_pred = log_fid.soft_predictions.transpose(
                "qec_round", "state", "shot"
            ).values
            leak_presence = ps_leak.leak_presence.transpose(
                "qec_round", "state", "shot"
            ).values.astype(int)
            no_leak_presence = 1 - leak_presence

            ps_log_errors = (log_errors * no_leak_presence).sum(axis=(1, 2))
            num_ps_shots = no_leak_presence.sum(axis=(1, 2))
            log_prob_leak = ps_log_errors / num_ps_shots

            soft_pred = soft_pred.reshape(len(log_fid.qec_round.values), -1)
            log_errors_ = log_errors.reshape(len(log_fid.qec_round.values), -1)
            soft_pred = 0.5 - np.abs(0.5 - soft_pred)
            order = np.sort(soft_pred, axis=1)
            log_prob_conf = np.zeros_like(log_prob_leak)
            for r, n in enumerate(num_ps_shots):
                thr = order[r, n]

                assert n >= (soft_pred[r] < thr).sum()
                log_prob_conf[r] = log_errors_[r][soft_pred[r] <= thr].mean()

            rounds = log_fid.qec_round.values
            assert (rounds == ROUNDS).all()

            i = 1 if "no_lru" in EXP_NAME else 0
            j = 1 if "nslru" in MODEL_NAME else 0
            k = k

            LOG_PROBS_LEAKAGE[i, j, k] = log_prob_leak
            LOG_PROBS_CONFIDENCE[i, j, k] = log_prob_conf
            NUM_SHOTS[i, j, k] = num_ps_shots

ds = xr.Dataset(
    data_vars=dict(
        log_probs_ps_leakage=(("lru", "ro", "leakage", "round"), LOG_PROBS_LEAKAGE),
        log_probs_ps_nn_confidence=(
            ("lru", "ro", "leakage", "round"),
            LOG_PROBS_CONFIDENCE,
        ),
        num_shots=(("lru", "ro", "leakage", "round"), NUM_SHOTS),
    ),
    coords=dict(lru=["yes", "no"], ro=["3ro", "2ro"], leakage=L1_RATES, round=ROUNDS),
)
ds.to_netcdf("post_selected_stability_experiment.nc")
