from copy import deepcopy

import numpy as np
import xarray as xr

from .preprocessing import to_inputs, to_model_input, to_inputs_stability


def evaluate_model(model, config, layout, injleak, dataset_name="test"):
    rounds = deepcopy(config.dataset[dataset_name]["rounds"])
    states = deepcopy(config.dataset[dataset_name]["states"])
    shots = deepcopy(config.dataset[dataset_name]["shots"])
    experiment_name = config.dataset["folder_format_name"]
    input_names = config.dataset["input_names"]
    data_type = int if "outcomes" in input_names else bool

    basis = config.dataset["basis"]
    stab_type = f"{basis.lower()}_type"

    dataset_dir = config.experiment_dir / dataset_name
    dataset_params = deepcopy(config.dataset[dataset_name])
    dataset_params["distance"] = deepcopy(config.dataset["distance"])
    for name in ["states", "rounds", "shots"]:
        dataset_params.pop(name)
    dataset_params["injleak"] = injleak

    # avoid error when num_rounds=0
    if 0 in rounds:
        rounds.pop(rounds.index(0))

    # compute num_shots if not specified (None)
    experiment = experiment_name.format(
        basis=basis,
        state=states[0],
        shots=shots,
        num_rounds=rounds[0],
        **dataset_params,
    )
    dataset = xr.open_dataset(dataset_dir / experiment / "measurements.nc")
    shots = len(dataset.shot)

    log_errors = np.zeros((len(rounds), len(states), shots), dtype=bool)
    soft_predictions = np.zeros((len(rounds), len(states), shots))
    correct_flips = np.zeros((len(rounds), len(states), shots), dtype=bool)
    for i, num_rounds in enumerate(rounds):
        for j, state in enumerate(states):
            experiment = experiment_name.format(
                basis=basis,
                state=state,
                shots=shots,
                num_rounds=num_rounds,
                **dataset_params,
            )
            dataset = xr.open_dataset(dataset_dir / experiment / "measurements.nc")
            proj_matrix = layout.projection_matrix(stab_type)

            # Convert to desired input
            dataset = to_inputs(dataset, proj_matrix, input_names=input_names)

            # Process for keras.model input
            dataset = to_model_input(*dataset, data_type=data_type)

            print(f"QEC = {num_rounds} | state = {state}", end="\r")
            soft_prediction = model.predict((dataset[0], dataset[1]), verbose=0)
            soft_prediction = soft_prediction[0].flatten()
            hard_prediction = soft_prediction > 0.5
            errors = hard_prediction != dataset[2]
            log_errors[i, j] = errors.astype(bool)
            soft_predictions[i, j] = soft_prediction
            correct_flips[i, j] = dataset[2].astype(bool)
            print(
                f"QEC = {num_rounds} | state = {state} | avg_errors = {np.average(errors):.4f}",
                end="\r",
            )

    log_fid = xr.Dataset(
        data_vars=dict(
            log_errors=(["qec_round", "state", "shot"], log_errors),
            soft_predictions=(["qec_round", "state", "shot"], soft_predictions),
            correct_flips=(["qec_round", "state", "shot"], correct_flips),
        ),
        coords=dict(qec_round=rounds, state=states, shot=list(range(shots))),
    )

    return log_fid


def evaluate_model_stability(model, config, layout, injleak, dataset_name="test"):
    rounds = deepcopy(config.dataset[dataset_name]["rounds"])
    states = deepcopy(config.dataset[dataset_name]["states"])
    shots = deepcopy(config.dataset[dataset_name]["shots"])
    experiment_name = config.dataset["folder_format_name"]
    input_names = config.dataset["input_names"]
    data_type = int if "outcomes" in input_names else bool

    basis = config.dataset["basis"]
    stab_type = f"{basis.lower()}_type"

    dataset_dir = config.experiment_dir / dataset_name
    dataset_params = deepcopy(config.dataset[dataset_name])
    dataset_params["distance"] = deepcopy(config.dataset["distance"])
    for name in ["states", "rounds", "shots"]:
        dataset_params.pop(name)
    dataset_params["injleak"] = injleak

    # avoid error when num_rounds=0,1
    if 0 in rounds:
        rounds.pop(rounds.index(0))
    if 1 in rounds:
        rounds.pop(rounds.index(1))

    # compute num_shots if not specified (None)
    experiment = experiment_name.format(
        basis=basis,
        state=states[0],
        shots=shots,
        num_rounds=rounds[0],
        **dataset_params,
    )
    dataset = xr.open_dataset(dataset_dir / experiment / "measurements.nc")
    shots = len(dataset.shot)

    log_errors = np.zeros((len(rounds), len(states), shots))
    soft_predictions = np.zeros((len(rounds), len(states), shots))
    correct_flips = np.zeros((len(rounds), len(states), shots), dtype=bool)
    for i, num_rounds in enumerate(rounds):
        for j, state in enumerate(states):
            experiment = experiment_name.format(
                basis=basis,
                state=state,
                shots=shots,
                num_rounds=num_rounds,
                **dataset_params,
            )
            dataset = xr.open_dataset(dataset_dir / experiment / "measurements.nc")

            # Convert to desired input
            dataset = to_inputs_stability(dataset, input_names=input_names)

            # Process for keras.model input
            dataset = to_model_input(*dataset, data_type=data_type)

            print(f"QEC = {num_rounds} | state = {state}", end="\r")
            soft_prediction = model.predict((dataset[0], dataset[1]), verbose=0)
            soft_prediction = soft_prediction.flatten()
            hard_prediction = soft_prediction > 0.5
            errors = hard_prediction != dataset[2]
            log_errors[i, j] = errors.astype(bool)
            soft_predictions[i, j] = soft_prediction
            correct_flips[i, j] = dataset[2].astype(bool)
            print(
                f"QEC = {num_rounds} | state = {state} | avg_errors = {np.average(errors):.4f}",
                end="\r",
            )

    log_fid = xr.Dataset(
        data_vars=dict(
            log_errors=(["qec_round", "state", "shot"], log_errors),
            soft_predictions=(["qec_round", "state", "shot"], soft_predictions),
            correct_flips=(["qec_round", "state", "shot"], correct_flips),
        ),
        coords=dict(qec_round=rounds, state=states, shot=list(range(shots))),
    )

    return log_fid
