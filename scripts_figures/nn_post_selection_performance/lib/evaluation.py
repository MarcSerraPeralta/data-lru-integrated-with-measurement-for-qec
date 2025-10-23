from copy import deepcopy

import numpy as np
import xarray as xr


def get_ps_leakage_memory(config, dataset_name="test"):
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

    ps_leak = np.zeros((len(rounds), len(states), shots), dtype=bool)
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
            anc_meas = dataset.anc_meas.transpose(
                "shot", "qec_round", "anc_qubit"
            ).values
            ps = (anc_meas == 2).any(axis=(1, 2))

            ps_leak[i, j] = ps

    ps_leak = xr.Dataset(
        data_vars=dict(
            leak_presence=(["qec_round", "state", "shot"], ps_leak),
        ),
        coords=dict(qec_round=rounds, state=states, shot=list(range(shots))),
    )

    return ps_leak


def get_ps_leakage_stability(config, dataset_name="test"):
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

    ps_leak = np.zeros((len(rounds), len(states), shots))
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
            anc_meas = dataset.anc_meas.transpose(
                "shot", "qec_round", "anc_qubit"
            ).values
            ps = (anc_meas == 2).any(axis=(1, 2))

            ps_leak[i, j] = ps

    ps_leak = xr.Dataset(
        data_vars=dict(
            leak_presence=(["qec_round", "state", "shot"], ps_leak),
        ),
        coords=dict(qec_round=rounds, state=states, shot=list(range(shots))),
    )

    return ps_leak
