import pathlib
import os

from qrennd import Config, Layout

from lib.evaluation import evaluate_model_stability
from lib.models import get_model

# Parameters
EXP_NAME = "20250619_stability7_exp_lru_v1"
TEST_DATASET = ["test"]

LAYOUT_NAME = "stability_a4_bZ.yaml"

OVERWRITE = False

PARENT_DIR = pathlib.Path.cwd()
DATA_DIR = PARENT_DIR / "data"
OUTPUT_DIR = PARENT_DIR / "output"

####################################


if not DATA_DIR.exists():
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")
if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

MODEL_NAMES = os.listdir(OUTPUT_DIR / EXP_NAME)
MODEL_NAMES = [n for n in MODEL_NAMES if n not in ["__old", ".DS_Store"]]
MODEL_NAMES = sorted(MODEL_NAMES)

for MODEL_NAME in MODEL_NAMES:
    MODEL_DIR = OUTPUT_DIR / EXP_NAME / MODEL_NAME
    if not MODEL_DIR.exists():
        raise ValueError(f"Model directory does not exist: {MODEL_DIR}")

    CONFIG_FILE = MODEL_DIR / "config.yaml"
    if not CONFIG_FILE.exists():
        raise ValueError(f"Config file does not exist: {CONFIG_FILE}")

    LAYOUT_FILE = DATA_DIR / EXP_NAME / "config" / LAYOUT_NAME
    if not LAYOUT_FILE.exists():
        raise ValueError(f"Layout file does not exist: {LAYOUT_FILE}")

    # check if the NN has finished training
    if not (MODEL_DIR / "checkpoint" / "final_weights.hdf5").exists():
        print(f"\nwarning: Model has not finished training: {MODEL_DIR}")

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
        print("")

        NAME = f"{test_dataset}.nc"
        if (MODEL_DIR / NAME).exists() and (not OVERWRITE):
            print("Model already evaluated!")
            print("output_dir=", OUTPUT_DIR)
            print("exp_name=", EXP_NAME)
            print("run_name=", MODEL_NAME)
            print("test_data=", test_dataset)
            continue

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

        if not (MODEL_DIR / "checkpoint" / "weights.keras").exists():
            print("ERROR: weights.keras does not exist")
            continue

        try:
            model.load_weights(MODEL_DIR / "checkpoint/weights.keras")
        except:
            print("ERROR: weights could not be loaded!")
            continue

        log_fid = evaluate_model_stability(model, config, layout, test_dataset)
        log_fid.to_netcdf(path=MODEL_DIR / NAME)

        print("Evaluation completed!")
        print("output_dir=", OUTPUT_DIR)
        print("exp_name=", EXP_NAME)
        print("run_name=", MODEL_NAME)
        print("test_data=", test_dataset)

