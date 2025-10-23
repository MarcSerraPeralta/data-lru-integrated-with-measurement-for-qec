import os
import pathlib
import warnings

warnings.filterwarnings("ignore")

from qrennd import Config, Layout
from lib.evaluation import evaluate_model_stability
from lib.models import get_model

# Parameters
TEST_DATASET = ["test"]
INJLEAKS = [0, 1, 2, 3, 4]
LAYOUT_NAME = "stability_a4_bZ.yaml"

OVERWRITE = True

PARENT_DIR = pathlib.Path.cwd()
DATA_DIR = PARENT_DIR / "data"
OUTPUT_DIR = PARENT_DIR / "output"

MODEL_NAMES = {
    "20250619_stability7_exp_lru_v1": [
        "20250904-144806_SDgB19_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-152916_4jqSEd_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-152916_7fG2sd_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-152916_xcbCBh_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-152919_muY3yQ_nslru_lstm32x4_eval32_b128_dr0-05_lr0-001_obs-end_injleak4",
        "20250902-195225_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250902-192211_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250902-233154_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250903-020745_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250903-084755_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250911-090803_uFPHdN_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250911-090803_aHModj_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250911-090803_Irq6q3_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250911-090803_nLw0DY_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250911-090803_tYYWaJ_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-223405_c4d3nB_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-223405_nHvynw_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-224337_LmTJQc_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-225507_KK2jYK_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-223405_VSZFV0_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250911-004148_QuOM8o_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-032750_c0VvC3_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-041255_L3ijU4_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-045751_VyMSgG_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-051528_0LErNm_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250908-101336_7hT2ZM_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250908-101336_Kfzfyd_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250908-101336_p3BM0D_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250908-101336_yQCNaT_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250908-101336_045SrX_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250909-094411_FbJssy_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-102410_3vGnVH_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-104645_BNLcJw_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-104645_BNLcJw_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-101756_KqfiGl_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250910-032018_lpdReg_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-032228_5rVX2o_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-032228_KuobDp_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-032228_RNsMBD_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-032228_laVx60_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-092547_koHD21_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-092602_3lg126_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-092809_L3Na9s_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-093019_ooj3Bt_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-092602_6Bt84N_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250911-145946_VrtLq0_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064035_E8lvzJ_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064035_TUywTa_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064842_YyX9KH_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-145946_f4TIgJ_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
    ],
    "20250619_stability7_exp_no_lru_v1": [
        "20250908-190538_Zzq7PV_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-191746_Ugkmyo_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-085047_EGxBrt_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-093017_X1tKgH_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-191746_6LcIYM_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250910-003637_hS7CUo_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-030808_G9JaeJ_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-030808_nQVw1E_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-031841_iIBuF1_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-030808_81eMXP_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-184846_UchCcZ_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-184846_zXsyiL_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-214031_7NFrbl_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-221413_c0sWAT_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-221413_u7BpWa_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250911-061539_IeEX12_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064035_OJTjZY_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-064035_zCZbsD_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-074311_a0w05L_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250916-113743_1HAT53_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250903-112208_HX8FwC_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250903-112315_8luiaZ_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250903-114158_RFLYFv_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250903-114158_V9jx4X_slru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250903-135613_b0sGDb_slru_lstm32x4_eval32_b128_dr0-05_lr0-003_obs-end_injleak4",
        "20250908-113455_HxJMKl_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-185826_JHI2nc_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-185826_OhpN7Z_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-185826_qOUJVE_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250908-101338_EK2MR7_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak0",
        "20250909-123618_lnDJja_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250909-183505_agfOWy_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250909-183505_VfXS9F_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250909-230154_sJ9mlK_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250909-231341_pbMAZO_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak1",
        "20250910-095952_QcxDvJ_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-095952_UXCTt4_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-100925_VGYPSE_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-154950_T8WITs_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250910-095952_j56Dtl_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak2",
        "20250911-074554_3bhrmY_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-074554_8G0sSy_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-074554_CZCB7B_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250911-074554_pU29hu_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250916-113743_1hTIJD_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak3",
        "20250904-115958_5v9vQ3_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-115958_HMv2tu_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250904-125030_3f0HXn_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250905-110640_kxxPrX_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
        "20250905-102650_sYADIs_nslru_lstm32x4_eval32_b128_dr0-05_lr0-002_obs-end_injleak4",
    ],
}

####################################

if not DATA_DIR.exists():
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")
if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

for exp_name, model_names in MODEL_NAMES.items():
    for model_name in model_names:
        MODEL_DIR = OUTPUT_DIR / exp_name / model_name
        if not MODEL_DIR.exists():
            raise ValueError(f"Model directory does not exist: {MODEL_DIR}")

        CONFIG_FILE = MODEL_DIR / "config.yaml"
        if not CONFIG_FILE.exists():
            raise ValueError(f"Config file does not exist: {CONFIG_FILE}")

        LAYOUT_FILE = DATA_DIR / exp_name / "config" / LAYOUT_NAME
        if not LAYOUT_FILE.exists():
            raise ValueError(f"Layout file does not exist: {LAYOUT_FILE}")

        # evaluate the NN
        layout = Layout.from_yaml(LAYOUT_FILE)
        config = Config.from_yaml(
            filepath=CONFIG_FILE,
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
        )
        if not isinstance(TEST_DATASET, list):
            TEST_DATASET = [TEST_DATASET]

        # if results have not been stored, evaluate model
        for test_dataset in TEST_DATASET:
            NAME = f"{test_dataset}.nc"

            print("Evaluating model...")
            print(MODEL_DIR / NAME)

            anc_qubits = layout.get_qubits(role="anc")
            num_anc = len(anc_qubits)

            rec_features = 0
            eval_features = 0
            if "outcomes" in config.dataset["input_names"]:
                rec_features += num_anc
                eval_features += num_anc
            if "binary_outcomes" in config.dataset["input_names"]:
                rec_features += num_anc
                eval_features += num_anc
            if "defects" in config.dataset["input_names"]:
                rec_features += num_anc
                eval_features += num_anc
            if "leakage_flags" in config.dataset["input_names"]:
                rec_features += num_anc
                eval_features += num_anc

            model = get_model(
                rec_features=rec_features,
                eval_features=eval_features,
                config=config,
            )

            # store model summary
            with open(MODEL_DIR / "model_summary.txt", "w") as file:
                model.summary(print_fn=lambda x: file.write(x + "\n"))

            weight_name = "weights.keras"
            if not (MODEL_DIR / "checkpoint" / weight_name).exists():
                print(
                    "ERROR: weights.keras does not exist, searching for weights from best epoch."
                )

                weights = [
                    f
                    for f in os.listdir(MODEL_DIR / "checkpoint")
                    if f.endswith(".keras")
                ]
                if len(weights) == 0:
                    continue

                weights = sorted(
                    weights, key=lambda x: float(x.replace(".keras", "").split("-")[2])
                )
                weight_name = weights[0]

            model.load_weights(MODEL_DIR / "checkpoint" / weight_name)

            for injleak in INJLEAKS:
                NAME_INJLEAK = f"{test_dataset}_injleak-{injleak}.nc"
                print("LEAKAGE", injleak, NAME_INJLEAK)
                if (MODEL_DIR / NAME_INJLEAK).exists() and (not OVERWRITE):
                    continue
                log_fid = evaluate_model_stability(
                    model, config, layout, injleak, test_dataset
                )
                log_fid.to_netcdf(path=MODEL_DIR / NAME_INJLEAK)

            print("Evaluation completed!")
            print("output_dir=", OUTPUT_DIR)
            print("exp_name=", exp_name)
            print("run_name=", model_name)
            print("test_data=", test_dataset)
