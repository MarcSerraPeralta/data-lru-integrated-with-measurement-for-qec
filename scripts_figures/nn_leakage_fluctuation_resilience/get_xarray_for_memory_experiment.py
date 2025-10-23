import pathlib

import numpy as np
import xarray as xr

ENSEMBLES_LRU = [
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
]

ENSEMBLES_NLRU = {
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
}

EXP_NAME_LRU = "20250718_repcode_d3_exp_lru_v2"
EXP_NAME_NLRU = "20250718_repcode_d3_exp_no_lru_v2"

INJLEAKS_L1 = [0.975, 1.575, 2.6, 3.625, 4.9]

PARENT_DIR = pathlib.Path.cwd()

# load data
ROUNDS = np.array(
    [1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 21, 26, 33, 42, 53, 68, 87, 108], dtype=int
)
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
ds.to_netcdf("l1_fluctuations_memory_experiment.nc")
