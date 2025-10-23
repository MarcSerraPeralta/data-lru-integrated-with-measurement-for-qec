import pathlib
import random
import numpy as np
import xarray as xr

from lib.layouts import stability
from lib.preprocessing import to_inputs_stability

# Parameters
LRU = "lru"  # ["lru", "no_lru"]
INJLEAK = 0  # [0, 1, 2, 3, 4]
NUM_ANC = 4
BASIS = "Z"

DC_DATA_DIR = pathlib.Path("...")
QRENND_DATA_DIR = f"data/20250619_stability7_exp_{LRU}_v1"

DATA = {
    "dir_name": "data",
    "file_name": "{LRU}_round_{num_rounds}__leak_{leak}_state_{state}.txt",
    "rounds": list(range(2, 40 + 1)),
    "states": list(range(0, 2**NUM_ANC)),
}
TEST_SAMPLES = 625  # test samples per state
VAL_FRACTION = 0.075

QRENND_DIR_NAME = "stability_a{num_anc}_r{num_rounds}_s{state}_b{basis}_injleak{leak}"
QRENND_FILE_NAME = "measurements.nc"

state_to_bitstring = lambda x: f"{x:0{NUM_ANC}b}"  # A0,A1,A2,... ordering

VERBOSE_STATE = True
VERBOSE_AVERAGE = True

###################

DC_DATA_DIR = pathlib.Path(DC_DATA_DIR)
if not DC_DATA_DIR.exists():
    raise ValueError(f"{DC_DATA_DIR} is not an existing directory.")

QRENND_DATA_DIR = pathlib.Path(QRENND_DATA_DIR)
QRENND_DATA_DIR.mkdir(exist_ok=True, parents=True)

QRENND_TRAIN_DIR = QRENND_DATA_DIR / "train"
QRENND_TRAIN_DIR.mkdir(exist_ok=True, parents=True)
QRENND_VAL_DIR = QRENND_DATA_DIR / "val"
QRENND_VAL_DIR.mkdir(exist_ok=True, parents=True)
QRENND_TEST_DIR = QRENND_DATA_DIR / "test"
QRENND_TEST_DIR.mkdir(exist_ok=True, parents=True)

QRENND_CONFIG_DIR = QRENND_DATA_DIR / "config"
QRENND_CONFIG_DIR.mkdir(exist_ok=True, parents=True)
layout = stability(NUM_ANC, BASIS)
layout.to_yaml(QRENND_CONFIG_DIR / f"stability_a{NUM_ANC}_b{BASIS}.yaml")

# process 'Train' data and split it to training and validation
for num_rounds in DATA["rounds"]:
    ave_defects = []
    ave_log_flips = []
    for state in DATA["states"]:
        # create file names and corresponding direcotires
        dc_file_name = DATA["file_name"].format(
            num_rounds=num_rounds,
            state=state,
            LRU=LRU,
            leak=INJLEAK,
        )
        qrennd_dir_name = QRENND_DIR_NAME.format(
            num_rounds=num_rounds,
            state=state_to_bitstring(state),
            basis=BASIS,
            leak=INJLEAK,
            num_anc=NUM_ANC,
        )
        (QRENND_TRAIN_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)
        (QRENND_VAL_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)
        (QRENND_TEST_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)

        # process DiCarlo data
        with open(DC_DATA_DIR / DATA["dir_name"] / dc_file_name, "r") as f:
            raw_data = f.read()

        outcomes = [
            list(map(int, line.split("\t")))
            for line in raw_data.split("\n")
            if line != ""
        ]
        outcomes = np.array(outcomes, dtype=int)

        num_shots, num_meas = outcomes.shape
        assert num_meas == NUM_ANC * num_rounds + NUM_ANC - 1

        _ = outcomes[:, -(NUM_ANC - 1) :]  # data measurements
        anc_meas = outcomes[:, : -(NUM_ANC - 1)]
        anc_meas = anc_meas.reshape(num_shots, NUM_ANC, num_rounds)
        anc_meas = anc_meas.swapaxes(1, 2)
        # anc_meas axes = (num_shots, num_rounds, anc_qubit)

        # ideal measurements for a single shot
        bitstring = np.array(list(map(int, state_to_bitstring(state))), dtype=bool)
        ideal_anc_meas = np.zeros((num_rounds, NUM_ANC), dtype=bool) ^ bitstring
        # the weight-1 ancillas get flipped due to the X echo gates
        ideal_anc_meas[1::4, 0] ^= True
        ideal_anc_meas[1::4, -1] ^= True
        ideal_anc_meas[::4, 0] ^= True
        ideal_anc_meas[::4, -1] ^= True

        # shuffle dataset because T1 fluctuates
        # suffle in a reproducible manner
        inds = list(range(num_shots))
        random.Random(123).shuffle(inds)
        anc_meas = anc_meas[inds]

        # split dataset into training, val and testing
        assert num_shots > TEST_SAMPLES

        nt = TEST_SAMPLES
        nv = int((num_shots - nt) * VAL_FRACTION)

        test_anc_meas = anc_meas[:nt]
        val_anc_meas = anc_meas[nt : nt + nv]
        train_anc_meas = anc_meas[nt + nv :]

        train_ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("shot", "qec_round", "anc_qubit"), train_anc_meas),
                ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
            ),
            coords=dict(
                shot=list(range(num_shots - nt - nv)),
                qec_round=list(range(1, num_rounds + 1)),
                anc_qubit=[f"A{i}" for i in range(NUM_ANC)],
                meas_reset=False,
                basis=BASIS,
            ),
        )
        train_ds.to_netcdf(QRENND_TRAIN_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        val_ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("shot", "qec_round", "anc_qubit"), val_anc_meas),
                ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
            ),
            coords=dict(
                shot=list(range(nv)),
                qec_round=list(range(1, num_rounds + 1)),
                anc_qubit=[f"A{i}" for i in range(NUM_ANC)],
                meas_reset=False,
                basis=BASIS,
            ),
        )
        val_ds.to_netcdf(QRENND_VAL_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        test_ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("shot", "qec_round", "anc_qubit"), test_anc_meas),
                ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
            ),
            coords=dict(
                shot=list(range(nt)),
                qec_round=list(range(1, num_rounds + 1)),
                anc_qubit=[f"A{i}" for i in range(NUM_ANC)],
                meas_reset=False,
                basis=BASIS,
            ),
        )
        test_ds.to_netcdf(QRENND_TEST_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        # print useful analysis
        initial_defect, defects, log_flips = to_inputs_stability(
            train_ds,
            input_names=["defects", "obs-definition-all"],
        )
        initial_defect = initial_defect[0].values.reshape(len(train_ds.shot), 1, -1)
        defects = defects[0].values
        defects = np.concatenate([initial_defect, defects], axis=1)

        ave_defects.append(defects.mean())
        ave_log_flips.append(log_flips.values.mean())

        if VERBOSE_STATE:
            for state in [0, 1, 2]:
                fraction = np.average(train_ds.anc_meas.values == state)
                print(f"{state}s fraction (train) = {fraction:0.3f}")

            print(dc_file_name)
            print(qrennd_dir_name)
            print(f"total number of training shots: {len(train_ds.shot)}")
            print(f"total number of validation shots: {len(val_ds.shot)}")
            print(f"total number of testing shots: {len(test_ds.shot)}")
            print(defects.mean(axis=0))
            print(log_flips.values.mean(axis=0))
            print(
                f"defect rates: {np.average(defects):0.4f} +/- {np.std(defects):0.4f}"
            )
            print(f"logical flips: {log_flips.mean():0.4f}")
            print("")

    if VERBOSE_AVERAGE:
        print(f"R = {num_rounds}")
        print(
            f"defect rates: {np.average(ave_defects):0.4f} +/- {np.std(ave_defects):0.4f}"
        )
        print(f"logical flips: {np.average(ave_log_flips):0.4f}")
        print("")
