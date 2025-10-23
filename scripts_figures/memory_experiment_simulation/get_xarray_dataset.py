import pathlib

import numpy as np
import xarray as xr

# Parameters
DATASETS = ["cz_sweep", "msmt_sweep", "1q_sweep", "resf_sweep", "fass_sweep"]
TEST_DATASET = "test"

PARENT_DIR = pathlib.Path.cwd()
DATA_DIR = PARENT_DIR / "data"
OUTPUT_DIR = PARENT_DIR / "output"

#############################

for dataset in DATASETS:
    # given by Yuejie
    if dataset == "cz_sweep":
        L1_RATES_LRU = L1_RATES_NLRU = np.array([0, 4, 8, 12, 16, 20])
        LEVELS_LRU = LEVELS_NLRU = [0, 1, 2, 3, 4, 5]
    elif dataset == "msmt_sweep":
        L1_RATES_LRU = L1_RATES_NLRU = np.linspace(0, 5, 6)
        LEVELS_LRU = LEVELS_NLRU = [0, 1, 2, 3, 4, 5]
    elif dataset == "1q_sweep":
        L1_RATES_LRU = L1_RATES_NLRU = np.array([0, 4, 8, 12, 16, 20])
        LEVELS_LRU = LEVELS_NLRU = [0, 1, 2, 3, 4, 5]
    elif dataset == "resf_sweep":
        L1_RATES_LRU = (
            np.linspace(0, 0.4, 6).tolist() + np.linspace(0.4, 0.8, 6)[1:].tolist()
        )
        LEVELS_LRU = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
        L1_RATES_NLRU = [0]
        LEVELS_NLRU = [0]
    elif dataset == "fass_sweep":
        L1_RATES_LRU = L1_RATES_NLRU = sorted(
            np.linspace(0, 0.4, 6).tolist() + np.linspace(0.04, 0.44, 6).tolist()
        )
        LEVELS_LRU = LEVELS_NLRU = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    else:
        raise ValueError(f"{dataset} not known")

    EXPERIMENTS = {
        f"20250930_repcode_d3_simulated_v3_{dataset}_txt_lru": {
            "SLRU": [
                f"20251002-000000_aaaaaa_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak{l}"
                for l in LEVELS_LRU
            ],
            "SLRU": [
                f"20251002-000000_aaaaaa_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak{l}"
                for l in LEVELS_LRU
            ],
        },
        f"20250930_repcode_d3_simulated_v3_{dataset}_txt_no_lru": {
            "SLRU": [
                f"20251002-000000_aaaaaa_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak{l}"
                for l in LEVELS_NLRU
            ],
            "SLRU": [
                f"20251002-000000_aaaaaa_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak{l}"
                for l in LEVELS_NLRU
            ],
        },
    }

    if not DATA_DIR.exists():
        raise ValueError(f"Data directory does not exist: {DATA_DIR}")
    if not OUTPUT_DIR.exists():
        raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

    ROUNDS = np.array(
        [1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 21, 26, 33, 42, 53, 68, 87, 108], dtype=int
    )
    LOG_PROBS = np.zeros(
        (2, 2, len(L1_RATES_LRU), len(ROUNDS))
    )  # lru, signaling, leakage, rounds
    for EXP_NAME, DATA in EXPERIMENTS.items():
        for NAME in DATA:
            for k, MODEL_NAME in enumerate(DATA[NAME]):
                MODEL_DIR = OUTPUT_DIR / EXP_NAME / MODEL_NAME
                if not MODEL_DIR.exists():
                    raise ValueError(f"Model directory does not exist: {MODEL_DIR}")

                NAME_NC = f"{TEST_DATASET}.nc"
                if not (MODEL_DIR / NAME_NC).exists():
                    print(f"No dataset found for {MODEL_DIR / NAME_NC}")
                    continue

                log_fid = xr.load_dataset(MODEL_DIR / NAME_NC)

                log_prob = log_fid.log_errors.transpose(
                    "qec_round", "state", "shot"
                ).values
                log_prob = log_prob.mean(axis=(1, 2))
                rounds = log_fid.qec_round.values
                assert (ROUNDS == rounds).all()

                i = 1 if "no_lru" in EXP_NAME else 0
                j = 1 if "NSLRU" in NAME else 0
                k = k
                LOG_PROBS[i, j, k] = log_prob

    ds = xr.Dataset(
        data_vars=dict(log_prob=(("lru", "ro", "l1", "round"), LOG_PROBS)),
        coords=dict(
            lru=["yes", "no"], ro=["3ro", "2ro"], l1=L1_RATES_LRU, round=ROUNDS
        ),
    )
    ds.to_netcdf(f"memory_experiment_results_simulation_{dataset}.nc")
