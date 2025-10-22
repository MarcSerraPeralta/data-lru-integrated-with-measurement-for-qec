import pathlib
import shutil

import numpy as np
import xarray as xr

# Parameters
EXP_NAME_LRU = "20250718_repcode_d3_exp_lru_v2"
EXP_NAME_NLRU = "20250718_repcode_d3_exp_no_lru_v2"
ENSEMBLES_LRU = ... # copy paste from f"ensembles/{EXP_NAME_LRU}.txt" file
ENSEMBLES_NLRU = ... # copy paste from f"ensembles/{EXP_NAME_NLRU}.txt" file

TEST_DATASET = "test"

OUTPUT_DIR = pathlib.Path.cwd() / "output"

####################################


if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

EXP_NAMES = [EXP_NAME_LRU, EXP_NAME_NLRU]
ENSEMBLES = [ENSEMBLES_LRU, ENSEMBLES_NLRU]
for exp_name, ensemble in zip(EXP_NAMES, ENSEMBLES):
    for ensemble_name, model_names in ensemble.items():
        ensemble_dir = OUTPUT_DIR / exp_name / ensemble_name
        ensemble_dir.mkdir(exist_ok=True, parents=True)

        soft_predictions = []
        correct_flips = None
        for model_name in model_names:
            model_dir = OUTPUT_DIR / exp_name / model_name
            if not model_dir.exists():
                raise ValueError(f"Model directory does not exist: {model_dir}")

            shutil.copy(model_dir / "config.yaml", ensemble_dir / "config.yaml")

            name = f"{TEST_DATASET}.nc"
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
            data_vars = dict(
                log_errors=(["qec_round", "state", "shot"], log_errors),
                soft_predictions=(["qec_round", "state", "shot"], ensemble_soft_predictions),
                correct_flips=(["qec_round", "state", "shot"], correct_flips),
            ),
            coords=dict(qec_round=log_fid.qec_round.values, 
                        state=log_fid.state.values,
                        shot=log_fid.shot.values,
                        ),
        )
        log_fid.to_netcdf(ensemble_dir / f"{TEST_DATASET}.nc")

        with open(ensemble_dir / "ensemble.txt", "w") as file:
            for model_name in model_names:
                model_dir = OUTPUT_DIR / exp_name / model_name
                file.write(str(model_dir) + "\n")
