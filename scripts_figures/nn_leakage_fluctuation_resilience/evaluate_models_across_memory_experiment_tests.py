import pathlib
import warnings

warnings.filterwarnings("ignore")

from qrennd import Config, Layout
from lib.evaluation import evaluate_model
from lib.models import get_model

# Parameters
TEST_DATASET = ["test"]
INJLEAKS = [0, 1, 2, 3, 4]
LAYOUT_NAME = "rep_code_d3_bZ.yaml"

OVERWRITE = False

PARENT_DIR = pathlib.Path.cwd()
DATA_DIR = PARENT_DIR / "data"
OUTPUT_DIR = PARENT_DIR / "output"

MODEL_NAMES = {
    "20250718_repcode_d3_exp_lru_v2": [
        "20250718-182433_slru_lstm24x2_eval24_b64_dr0-10_lr0-001",
        "20250725-211221_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-211547_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-211756_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-212110_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250722-112606_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-203717_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-204613_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-205221_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-205426_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250722-112820_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-201641_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-202047_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-202256_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-202605_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250722-113012_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-195514_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-200647_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-200914_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-201227_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250718-191853_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-193700_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-193924_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-194027_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-194130_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250719-043805_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250726-133950_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250727-120801_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250728-151433_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250728-155159_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250722-113826_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250726-133336_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250726-133542_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250727-121516_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250729-113112_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250722-113427_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250726-123701_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-122118_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-122621_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250729-114727_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250722-113207_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250726-125428_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-124022_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-124842_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-125636_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250720-191541_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250726-132036_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-130540_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-131444_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-132248_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
    ],
    "20250718_repcode_d3_exp_no_lru_v2": [
        "20250718-193534_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-155755_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-155817_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-160351_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250725-161151_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250722-113820_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-160542_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-160636_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-160843_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250725-160947_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250722-113433_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-161356_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-162343_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-172537_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250725-172859_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250722-113205_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-173001_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-173314_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-173624_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250725-175426_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250718-192410_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-175941_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-181308_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-184123_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250725-191658_slru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250718-234626_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250729-121553_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250729-145129_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250730-015841_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250730-223350_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak0",
        "20250722-112544_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250731-105118_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250731-133550_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250731-135525_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250731-140453_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak1",
        "20250722-112826_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-162311_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-164138_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-170819_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250727-171250_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak2",
        "20250722-112956_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-140109_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-140721_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-142627_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250727-155956_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak3",
        "20250721-045448_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-133152_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-134104_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-134803_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
        "20250727-135403_nslru_lstm24x2_eval24_b64_dr0-10_lr0-001_injleak4",
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
            for injleak in INJLEAKS:
                NAME = f"{test_dataset}_injleak-{injleak}.nc"

                print("Evaluating model...")
                print(MODEL_DIR / NAME)
                if (MODEL_DIR / NAME).exists() and (not OVERWRITE):
                    continue

                anc_qubits = layout.get_qubits(role="anc")
                num_anc = len(anc_qubits)

                rec_features = 0
                eval_features = num_anc
                if "outcomes" in config.dataset["input_names"]:
                    rec_features += num_anc
                if "binary_outcomes" in config.dataset["input_names"]:
                    rec_features += num_anc
                if "defects" in config.dataset["input_names"]:
                    rec_features += num_anc
                if "leakage_flags" in config.dataset["input_names"]:
                    rec_features += num_anc

                model = get_model(
                    rec_features=rec_features,
                    eval_features=eval_features,
                    config=config,
                )

                if not (MODEL_DIR / "checkpoint" / "weights.keras").exists():
                    raise ValueError("ERROR: weights.keras does not exist")

                model.load_weights(MODEL_DIR / "checkpoint/weights.keras")
                log_fid = evaluate_model(model, config, layout, injleak, test_dataset)
                log_fid.to_netcdf(path=MODEL_DIR / NAME)

                print("Evaluation completed!")
                print("output_dir=", OUTPUT_DIR)
                print("exp_name=", exp_name)
                print("run_name=", model_name)
                print("test_data=", test_dataset)
                print("injleak=", injleak)
