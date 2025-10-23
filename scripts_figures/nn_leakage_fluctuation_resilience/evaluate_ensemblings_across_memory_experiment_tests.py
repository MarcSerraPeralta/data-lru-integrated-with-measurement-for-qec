import pathlib
import os

import numpy as np
import xarray as xr

# Parameters

# memory experiments
ENSEMBLES_LRU = {
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0": [
        "20250718-182433_slru_lstm24x2_eval24_b64_dr0-10_lr0-001",
        "20250725-211221_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-211547_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-211756_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-212110_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
    ],
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1": [
        "20250722-112606_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-203717_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-204613_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-205221_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-205426_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
    ],
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2": [
        "20250722-112820_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-201641_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-202047_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-202256_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-202605_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
    ],
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3": [
        "20250722-113012_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-195514_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-200647_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-200914_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-201227_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
    ],
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4": [
        "20250718-191853_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-193700_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-193924_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-194027_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-194130_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0": [
        "20250719-043805_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250726-133950_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250727-120801_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250728-151433_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250728-155159_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1": [
        "20250722-113826_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250726-133336_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250726-133542_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250727-121516_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250729-113112_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2": [
        "20250722-113427_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250726-123701_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-122118_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-122621_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250729-114727_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3": [
        "20250722-113207_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250726-125428_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-124022_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-124842_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-125636_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4": [
        "20250720-191541_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250726-132036_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-130540_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-131444_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-132248_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
    ],
}

ENSEMBLES_NLRU = {
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0": [
        "20250718-193534_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-155755_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-155817_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-160351_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-161151_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
    ],
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1": [
        "20250722-113820_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-160542_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-160636_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-160843_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-160947_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
    ],
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2": [
        "20250722-113433_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-161356_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-162343_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-172537_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-172859_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
    ],
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3": [
        "20250722-113205_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-173001_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-173314_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-173624_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-175426_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
    ],
    "20250728-000000_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4": [
        "20250718-192410_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-175941_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-181308_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-184123_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-191658_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0": [
        "20250718-234626_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250729-121553_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250729-145129_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250730-015841_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250730-223350_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1": [
        "20250722-112544_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250731-105118_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250731-133550_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250731-135525_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250731-140453_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2": [
        "20250722-112826_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-162311_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-164138_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-170819_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-171250_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3": [
        "20250722-112956_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-140109_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-140721_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-142627_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-155956_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
    ],
    "20250728-000000_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4": [
        "20250721-045448_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-133152_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-134104_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-134803_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-135403_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
    ],
}


EXP_NAME_LRU = "20250718_repcode_d3_exp_lru_v2"
EXP_NAME_NLRU = "20250718_repcode_d3_exp_no_lru_v2"

TEST_DATASET = "test"
INJLEAKS = [0, 1, 2, 3, 4]

PARENT_DIR = pathlib.Path.cwd()
OUTPUT_DIR = PARENT_DIR / "output"

####################################


if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

for injleak in INJLEAKS:
    for ensemble_name, model_names in ENSEMBLES_LRU.items():
        ensemble_dir = OUTPUT_DIR / EXP_NAME_LRU / ensemble_name
        ensemble_dir.mkdir(exist_ok=True, parents=True)

        soft_predictions = []
        correct_flips = None
        for model_name in model_names:
            model_dir = OUTPUT_DIR / EXP_NAME_LRU / model_name
            if not model_dir.exists():
                raise ValueError(f"Model directory does not exist: {model_dir}")

            name = f"{TEST_DATASET}_injleak-{injleak}.nc"
            if not (model_dir / name).exists():
                raise ValueError(f"{model_dir / name} does not exist.")

            log_fid = xr.load_dataset(model_dir / name)
            soft_predictions.append(log_fid.soft_predictions)

            if correct_flips is None:
                correct_flips = log_fid.correct_flips

            assert (correct_flips == log_fid.correct_flips).all()

        # geometric mean
        ensemble_soft_predictions = 0
        for soft_prediction in soft_predictions:
            ensemble_soft_predictions += np.log(soft_prediction.values)
        ensemble_soft_predictions /= len(soft_predictions)
        ensemble_soft_predictions = np.exp(ensemble_soft_predictions)

        correct_flips = correct_flips.values
        ensemble_hard_predictions = ensemble_soft_predictions > 0.5
        log_errors = ensemble_hard_predictions != correct_flips

        log_fid = xr.Dataset(
            data_vars=dict(
                log_errors=(["qec_round", "state", "shot"], log_errors),
                soft_predictions=(
                    ["qec_round", "state", "shot"],
                    ensemble_soft_predictions,
                ),
                correct_flips=(["qec_round", "state", "shot"], correct_flips),
            ),
            coords=dict(
                qec_round=log_fid.qec_round.values,
                state=log_fid.state.values,
                shot=log_fid.shot.values,
            ),
        )
        log_fid.to_netcdf(ensemble_dir / f"{TEST_DATASET}_injleak-{injleak}.nc")

        with open(ensemble_dir / f"ensemble_injleak-{injleak}.txt", "w") as file:
            for model_name in model_names:
                model_dir = OUTPUT_DIR / EXP_NAME_LRU / model_name
                file.write(str(model_dir) + "\n")


for injleak in INJLEAKS:
    for ensemble_name, model_names in ENSEMBLES_NLRU.items():
        ensemble_dir = OUTPUT_DIR / EXP_NAME_NLRU / ensemble_name
        ensemble_dir.mkdir(exist_ok=True, parents=True)

        soft_predictions = []
        correct_flips = None
        for model_name in model_names:
            model_dir = OUTPUT_DIR / EXP_NAME_NLRU / model_name
            if not model_dir.exists():
                raise ValueError(f"Model directory does not exist: {model_dir}")

            name = f"{TEST_DATASET}_injleak-{injleak}.nc"
            if not (model_dir / name).exists():
                raise ValueError(f"{model_dir / name} does not exist.")

            log_fid = xr.load_dataset(model_dir / name)
            soft_predictions.append(log_fid.soft_predictions)

            if correct_flips is None:
                correct_flips = log_fid.correct_flips

            assert (correct_flips == log_fid.correct_flips).all()

        # geometric mean
        ensemble_soft_predictions = 0
        for soft_prediction in soft_predictions:
            ensemble_soft_predictions += np.log(soft_prediction.values)
        ensemble_soft_predictions /= len(soft_predictions)
        ensemble_soft_predictions = np.exp(ensemble_soft_predictions)

        correct_flips = correct_flips.values
        ensemble_hard_predictions = ensemble_soft_predictions > 0.5
        log_errors = ensemble_hard_predictions != correct_flips

        log_fid = xr.Dataset(
            data_vars=dict(
                log_errors=(["qec_round", "state", "shot"], log_errors),
                soft_predictions=(
                    ["qec_round", "state", "shot"],
                    ensemble_soft_predictions,
                ),
                correct_flips=(["qec_round", "state", "shot"], correct_flips),
            ),
            coords=dict(
                qec_round=log_fid.qec_round.values,
                state=log_fid.state.values,
                shot=log_fid.shot.values,
            ),
        )
        log_fid.to_netcdf(ensemble_dir / f"{TEST_DATASET}_injleak-{injleak}.nc")

        with open(ensemble_dir / f"ensemble_injleak-{injleak}.txt", "w") as file:
            for model_name in model_names:
                model_dir = OUTPUT_DIR / EXP_NAME_NLRU / model_name
                file.write(str(model_dir) + "\n")
