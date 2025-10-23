import pathlib

import numpy as np
import xarray as xr

ENSEMBLES_LRU = [
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
]

ENSEMBLES_NLRU = [
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
]

EXP_NAME_LRU = "20250619_stability7_exp_lru_v1"
EXP_NAME_NLRU = "20250619_stability7_exp_no_lru_v1"

INJLEAKS_L1 = [1.2, 2.8, 3.7, 5.3, 7.6]  # given by Yuejie

PARENT_DIR = pathlib.Path.cwd()

# load data
ROUNDS = np.arange(3, 41, dtype=int)
LOG_PROBS = np.zeros(
    (2, 2, 5, 5, len(ROUNDS))
)  # lru, signaling, train leakage, test leakage, rounds
for test_injleak in range(5):
    for name in ENSEMBLES_LRU:
        nn_injleak = int(name[-1])
        nn_index = 1 if "nslru" in name else 0

        data = xr.load_dataset(
            PARENT_DIR / EXP_NAME_LRU / name / f"test_injleak-{test_injleak}.nc"
        )

        log_prob = data.log_errors.transpose("qec_round", "state", "shot").values
        log_prob = log_prob.mean(axis=(1, 2))
        rounds = data.qec_round.values
        assert (rounds == ROUNDS).all()

        LOG_PROBS[0, nn_index, nn_injleak, test_injleak] = log_prob

    for name in ENSEMBLES_NLRU:
        nn_injleak = int(name[-1])
        nn_index = 1 if "nslru" in name else 0

        data = xr.load_dataset(
            PARENT_DIR / EXP_NAME_NLRU / name / f"test_injleak-{test_injleak}.nc"
        )

        log_prob = data.log_errors.transpose("qec_round", "state", "shot").values
        log_prob = log_prob.mean(axis=(1, 2))
        rounds = data.qec_round.values
        assert (rounds == ROUNDS).all()

        LOG_PROBS[1, nn_index, nn_injleak, test_injleak] = log_prob

ds = xr.Dataset(
    data_vars=dict(log_prob=(("lru", "ro", "train_l1", "test_l1", "round"), LOG_PROBS)),
    coords=dict(
        lru=["yes", "no"],
        ro=["3ro", "2ro"],
        train_l1=INJLEAKS_L1,
        test_l1=INJLEAKS_L1,
        round=ROUNDS,
    ),
)
ds.to_netcdf("l1_fluctuations_stability_experiment.nc")
