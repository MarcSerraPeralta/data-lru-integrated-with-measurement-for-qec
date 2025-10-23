import pathlib
import random
import numpy as np
import xarray as xr

from lib.layouts import rep_code
from lib.preprocessing import to_inputs

# Parameters
LRU = "lru"
INJLEAK = 3
DISTANCE = 3
BASIS = "Z"

DC_DATA_DIR = pathlib.Path("...")
QRENND_DATA_DIR = f"data/20250718_repcode_d3_exp_{LRU}_v2_cheating"

DATA_TRAIN = {
    "dir_name": "train",
    "file_name": "{lru}_round_{num_rounds}_leak_{injleak}_state_{state}.txt",
    "rounds": list(np.concatenate([np.arange(1, 8, 1), np.arange(8, 43, 3)])),
    "states": list(range(0, 2**3)),
}
DATA_TEST = {
    "dir_name": "test",
    "file_name": "{lru}_round_{num_rounds}_leak_{injleak}_state_{state}.txt",
    "rounds": [1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 21, 26, 33, 42, 53, 68, 87, 108],
    "states": list(range(0, 2**3)),
}
VAL_FRACTION = 0.10
TRAIN_SHOTS = 9000

QRENND_DIR_NAME = (
    "repcode_d{distance}_r{num_rounds}_s{state}_b{basis}_injleak{injleak}_cheat{cheat}"
)
QRENND_FILE_NAME = "measurements.nc"

# they gave me a dataset with opposite ordering as before!
state_to_bitstring = lambda x: f"{x:0{DISTANCE}b}"

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
layout = rep_code(DISTANCE, BASIS)
layout.to_yaml(QRENND_CONFIG_DIR / f"rep_code_d{DISTANCE}_b{BASIS}.yaml")

# process 'Train' data and split it to training and validation
for num_rounds in DATA_TRAIN["rounds"]:
    for state in DATA_TRAIN["states"]:
        # create file names and corresponding direcotires
        dc_file_name = DATA_TRAIN["file_name"].format(
            num_rounds=num_rounds,
            state=state,  # state_to_bitstring(state),
            lru=LRU,
            injleak=INJLEAK,
        )
        qrennd_dir_name = QRENND_DIR_NAME.format(
            distance=DISTANCE,
            num_rounds=num_rounds,
            state=state_to_bitstring(state),
            basis=BASIS,
            injleak=INJLEAK,
            cheat=False,
        )
        (QRENND_TRAIN_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)
        (QRENND_VAL_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)

        # process DiCarlo data
        with open(DC_DATA_DIR / DATA_TRAIN["dir_name"] / dc_file_name, "r") as f:
            raw_data = f.read()

        outcomes = [
            list(map(int, line.split("\t")))
            for line in raw_data.split("\n")
            if line != ""
        ]
        outcomes = np.array(outcomes, dtype=int)

        num_shots, num_meas = outcomes.shape
        assert num_meas == (DISTANCE - 1) * num_rounds + DISTANCE

        data_meas = outcomes[:, -DISTANCE:]
        anc_meas = outcomes[:, :-DISTANCE]
        anc_meas = anc_meas.reshape(num_shots, DISTANCE - 1, num_rounds)
        anc_meas = anc_meas.swapaxes(1, 2)
        # anc_meas axes = (num_shots, num_rounds, anc_qubit)

        # ideal measurements for a single shot
        bitstring = np.array(list(map(int, state_to_bitstring(state))), dtype=bool)
        projected_bitstring = np.array(
            [i ^ j for i, j in zip(bitstring[:-1], bitstring[1:])], dtype=bool
        )
        ideal_anc_meas = (
            np.zeros((num_rounds, DISTANCE - 1), dtype=bool) ^ projected_bitstring
        )
        ideal_anc_meas[1::2] ^= projected_bitstring  # no resets in ancillas
        ideal_data_meas = np.zeros_like(DISTANCE, dtype=bool) ^ bitstring
        if (num_rounds - 1) % 2 == 1:
            ideal_data_meas ^= True  # echo gates

        # shuffle dataset because T1 fluctuates
        # suffle in a reproducible manner
        inds = list(range(num_shots))
        random.Random(123).shuffle(inds)
        # remove train 'non-cheating' shots so that it matches the number of
        # train 'cheating' shots
        assert num_shots >= TRAIN_SHOTS
        inds = inds[:TRAIN_SHOTS]
        num_shots = TRAIN_SHOTS
        anc_meas, data_meas = anc_meas[inds], data_meas[inds]

        # split dataset into training and val
        nv = int((num_shots) * VAL_FRACTION)

        val_anc_meas, val_data_meas = anc_meas[:nv], data_meas[:nv]
        train_anc_meas, train_data_meas = anc_meas[nv:], data_meas[nv:]

        train_ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("shot", "qec_round", "anc_qubit"), train_anc_meas),
                ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
                data_meas=(("shot", "data_qubit"), train_data_meas),
                ideal_data_meas=(("data_qubit"), ideal_data_meas),
            ),
            coords=dict(
                shot=list(range(num_shots - nv)),
                qec_round=list(range(1, num_rounds + 1)),
                data_qubit=[f"D{i}" for i in range(DISTANCE)],
                anc_qubit=[f"A{i}" for i in range(DISTANCE - 1)],
                meas_reset=False,
                basis=BASIS,
            ),
        )
        train_ds.to_netcdf(QRENND_TRAIN_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        val_ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("shot", "qec_round", "anc_qubit"), val_anc_meas),
                ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
                data_meas=(("shot", "data_qubit"), val_data_meas),
                ideal_data_meas=(("data_qubit"), ideal_data_meas),
            ),
            coords=dict(
                shot=list(range(nv)),
                qec_round=list(range(1, num_rounds + 1)),
                data_qubit=[f"D{i}" for i in range(DISTANCE)],
                anc_qubit=[f"A{i}" for i in range(DISTANCE - 1)],
                meas_reset=False,
                basis=BASIS,
            ),
        )
        val_ds.to_netcdf(QRENND_VAL_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        # display useful analysis
        print(qrennd_dir_name)
        print(f"total number of training shots: {len(train_ds.shot)}")
        print(f"total number of validation shots: {len(val_ds.shot)}")
        for state in [0, 1, 2]:
            fraction = np.average(train_ds.anc_meas.values == state)
            print(f"{state}s fraction (train) = {fraction:0.3f}")

        proj_matrix = layout.projection_matrix(f"{BASIS.lower()}_type")
        defects, final_defects, log_flips = to_inputs(
            train_ds, proj_matrix, input_names=["defects"]
        )
        defects = defects[0].mean(dim="shot")
        final_defects = final_defects.mean(dim="shot")
        defects = np.concatenate(
            [defects.values, final_defects.values.reshape(1, -1)], axis=0
        )
        print(f"defect rates: {np.average(defects):0.4f} +/- {np.std(defects):0.4f}")
        print(f"logical flips: {log_flips.mean():0.4f}")
        print("")

# process 'Test' data
for num_rounds in DATA_TEST["rounds"]:
    for state in DATA_TEST["states"]:
        # create file names and corresponding direcotires
        dc_file_name = DATA_TEST["file_name"].format(
            num_rounds=num_rounds,
            state=state,  # state_to_bitstring(state),
            lru=LRU,
            injleak=INJLEAK,
        )
        qrennd_dir_name = QRENND_DIR_NAME.format(
            distance=DISTANCE,
            num_rounds=num_rounds,
            state=state_to_bitstring(state),
            basis=BASIS,
            injleak=INJLEAK,
            cheat=True,
        )
        (QRENND_TEST_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)
        (QRENND_TRAIN_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)
        (QRENND_VAL_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)

        # process DiCarlo data
        with open(DC_DATA_DIR / DATA_TEST["dir_name"] / dc_file_name, "r") as f:
            raw_data = f.read()

        outcomes = [
            list(map(int, line.split("\t")))
            for line in raw_data.split("\n")
            if line != ""
        ]
        outcomes = np.array(outcomes, dtype=int)

        num_shots, num_meas = outcomes.shape
        assert num_meas == (DISTANCE - 1) * num_rounds + DISTANCE

        data_meas = outcomes[:, -DISTANCE:]
        anc_meas = outcomes[:, :-DISTANCE]
        anc_meas = anc_meas.reshape(num_shots, DISTANCE - 1, num_rounds)
        anc_meas = anc_meas.swapaxes(1, 2)
        # anc_meas axes = (num_shots, num_rounds, anc_qubit)

        # ideal measurements for a single shot
        bitstring = np.array(list(map(int, state_to_bitstring(state))), dtype=bool)
        projected_bitstring = np.array(
            [i ^ j for i, j in zip(bitstring[:-1], bitstring[1:])], dtype=bool
        )
        ideal_anc_meas = (
            np.zeros((num_rounds, DISTANCE - 1), dtype=bool) ^ projected_bitstring
        )
        ideal_anc_meas[1::2] ^= projected_bitstring  # no resets in ancillas
        ideal_data_meas = np.zeros_like(DISTANCE, dtype=bool) ^ bitstring
        if (num_rounds - 1) % 2 == 1:
            ideal_data_meas ^= True  # echo gates

        # shuffle dataset because T1 fluctuates
        # suffle in a reproducible manner
        inds = list(range(num_shots))
        random.Random(123).shuffle(inds)
        anc_meas, data_meas = anc_meas[inds], data_meas[inds]

        # split test dataset into train 'cheating' and test
        assert num_shots > TRAIN_SHOTS
        nt, nv = int(TRAIN_SHOTS * (1 - VAL_FRACTION)), int(TRAIN_SHOTS * VAL_FRACTION)

        train_anc_meas, train_data_meas = anc_meas[:nt], data_meas[:nt]
        val_anc_meas, val_data_meas = anc_meas[nt : nv + nt], data_meas[nt : nv + nt]
        test_anc_meas, test_data_meas = anc_meas[nv + nt :], data_meas[nv + nt :]

        # try to make the rounds in train 'cheating' and train 'non-cheating' as
        # close as possible by removing rounds from train 'cheating'
        max_round = max(DATA_TRAIN["rounds"])
        if num_rounds <= max_round + 2:
            train_ds = xr.Dataset(
                data_vars=dict(
                    anc_meas=(("shot", "qec_round", "anc_qubit"), train_anc_meas),
                    ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
                    data_meas=(("shot", "data_qubit"), train_data_meas),
                    ideal_data_meas=(("data_qubit"), ideal_data_meas),
                ),
                coords=dict(
                    shot=list(range(nt)),
                    qec_round=list(range(1, num_rounds + 1)),
                    data_qubit=[f"D{i}" for i in range(DISTANCE)],
                    anc_qubit=[f"A{i}" for i in range(DISTANCE - 1)],
                    meas_reset=False,
                    basis=BASIS,
                ),
            )
            train_ds.to_netcdf(QRENND_TRAIN_DIR / qrennd_dir_name / QRENND_FILE_NAME)

            val_ds = xr.Dataset(
                data_vars=dict(
                    anc_meas=(("shot", "qec_round", "anc_qubit"), val_anc_meas),
                    ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
                    data_meas=(("shot", "data_qubit"), val_data_meas),
                    ideal_data_meas=(("data_qubit"), ideal_data_meas),
                ),
                coords=dict(
                    shot=list(range(nv)),
                    qec_round=list(range(1, num_rounds + 1)),
                    data_qubit=[f"D{i}" for i in range(DISTANCE)],
                    anc_qubit=[f"A{i}" for i in range(DISTANCE - 1)],
                    meas_reset=False,
                    basis=BASIS,
                ),
            )
            val_ds.to_netcdf(QRENND_VAL_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        test_ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("shot", "qec_round", "anc_qubit"), test_anc_meas),
                ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
                data_meas=(("shot", "data_qubit"), test_data_meas),
                ideal_data_meas=(("data_qubit"), ideal_data_meas),
            ),
            coords=dict(
                shot=list(range(num_shots - nv - nt)),
                qec_round=list(range(1, num_rounds + 1)),
                data_qubit=[f"D{i}" for i in range(DISTANCE)],
                anc_qubit=[f"A{i}" for i in range(DISTANCE - 1)],
                meas_reset=False,
                basis=BASIS,
            ),
        )
        test_ds.to_netcdf(QRENND_TEST_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        # display useful analysis
        print(qrennd_dir_name)
        if num_rounds <= max_round + 2:
            print(f"total number of train shots: {len(train_ds.shot)}")
            print(f"total number of val shots: {len(val_ds.shot)}")
        print(f"total number of test shots: {len(test_ds.shot)}")
        for state in [0, 1, 2]:
            fraction = np.average(test_ds.anc_meas.values == state)
            print(f"{state}s fraction (test) = {fraction:0.3f}")

        proj_matrix = layout.projection_matrix(f"{BASIS.lower()}_type")
        defects, final_defects, log_flips = to_inputs(
            test_ds, proj_matrix, input_names=["defects"]
        )
        defects = defects[0].mean(dim="shot")
        final_defects = final_defects.mean(dim="shot")
        defects = np.concatenate(
            [defects.values, final_defects.values.reshape(1, -1)], axis=0
        )

        print(
            f"defect rates (test): {np.average(defects):0.4f} +/- {np.std(defects):0.4f}"
        )
        print(f"logical flips (test): {log_flips.mean():0.4f}")
        print("")
