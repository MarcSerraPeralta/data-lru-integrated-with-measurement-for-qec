import pathlib
import os

import numpy as np
import xarray as xr

# Parameters

# stability experiments
ENSEMBLES_LRU = {
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0": [
        "20250902-195225_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250902-192211_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250902-233154_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250903-020745_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250903-084755_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
    ],
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1": [
        "20250911-090803_uFPHdN_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250911-090803_aHModj_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250911-090803_Irq6q3_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250911-090803_nLw0DY_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250911-090803_tYYWaJ_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
    ],
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2": [
        "20250910-223405_c4d3nB_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-223405_nHvynw_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-224337_LmTJQc_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-225507_KK2jYK_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-223405_VSZFV0_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
    ],
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3": [
        "20250911-004148_QuOM8o_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-032750_c0VvC3_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-041255_L3ijU4_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-045751_VyMSgG_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-051528_0LErNm_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
    ],
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4": [
        "20250908-101336_7hT2ZM_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250908-101336_Kfzfyd_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250908-101336_p3BM0D_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250908-101336_yQCNaT_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250908-101336_045SrX_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0": [
        "20250909-094411_FbJssy_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-102410_3vGnVH_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-104645_BNLcJw_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-104645_BNLcJw_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-101756_KqfiGl_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1": [
        "20250910-032018_lpdReg_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-032228_5rVX2o_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-032228_KuobDp_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-032228_RNsMBD_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-032228_laVx60_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2": [
        "20250910-092547_koHD21_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-092602_3lg126_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-092809_L3Na9s_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-093019_ooj3Bt_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-092602_6Bt84N_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3": [
        "20250911-145946_VrtLq0_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064035_E8lvzJ_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064035_TUywTa_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064842_YyX9KH_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-145946_f4TIgJ_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4": [
        "20250904-144806_SDgB19_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-152916_4jqSEd_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-152916_7fG2sd_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-152916_xcbCBh_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-152919_muY3yQ_nslru_lstm32x4_eval32_b128_dr0-05_lr0-001_obs-end_injleak4",
    ],
}

ENSEMBLES_NLRU = {
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0": [
        "20250908-190538_Zzq7PV_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-191746_Ugkmyo_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-085047_EGxBrt_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-093017_X1tKgH_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-191746_6LcIYM_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
    ],
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1": [
        "20250910-003637_hS7CUo_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-030808_G9JaeJ_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-030808_nQVw1E_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-031841_iIBuF1_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-030808_81eMXP_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
    ],
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2": [
        "20250910-184846_UchCcZ_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-184846_zXsyiL_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-214031_7NFrbl_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-221413_c0sWAT_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-221413_u7BpWa_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
    ],
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3": [
        "20250911-061539_IeEX12_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064035_OJTjZY_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064035_zCZbsD_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-074311_a0w05L_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250916-113743_1HAT53_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
    ],
    "20250903-000000_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4": [
        "20250903-112208_HX8FwC_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250903-112315_8luiaZ_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250903-114158_RFLYFv_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250903-114158_V9jx4X_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250903-135613_b0sGDb_slru_lstm32x4_eval32_b128_dr0-05_lr0-003_obs-end_injleak4",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0": [
        "20250908-113455_HxJMKl_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-185826_JHI2nc_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-185826_OhpN7Z_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-185826_qOUJVE_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-101338_EK2MR7_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1": [
        "20250909-123618_lnDJja_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250909-183505_agfOWy_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250909-183505_VfXS9F_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250909-230154_sJ9mlK_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250909-231341_pbMAZO_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2": [
        "20250910-095952_QcxDvJ_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-095952_UXCTt4_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-100925_VGYPSE_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-154950_T8WITs_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-095952_j56Dtl_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3": [
        "20250911-074554_3bhrmY_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-074554_8G0sSy_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-074554_CZCB7B_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-074554_pU29hu_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250916-113743_1hTIJD_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
    ],
    "20250903-000000_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4": [
        "20250904-115958_5v9vQ3_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-115958_HMv2tu_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-125030_3f0HXn_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250905-110640_kxxPrX_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250905-102650_sYADIs_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
    ],
}


EXP_NAME_LRU = "20250619_stability7_exp_lru_v1"
EXP_NAME_NLRU = "20250619_stability7_exp_no_lru_v1"

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

        if (ensemble_dir / f"{TEST_DATASET}_injleak-{injleak}.nc").exists():
            continue

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

        if (ensemble_dir / f"{TEST_DATASET}_injleak-{injleak}.nc").exists():
            continue

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
