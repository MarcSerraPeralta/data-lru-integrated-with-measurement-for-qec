from qrennd.configs import Config
from qrennd.layouts import Layout

from .generators import dataset_generator
from .preprocessing import (
    to_model_input,
    to_inputs,
    to_inputs_stability,
    add_classical_meas_noise,
)


def load_datasets(config: Config, layout: Layout, dataset_name: str):
    batch_size = config.train["batch_size"]
    experiment_name = config.dataset["folder_format_name"]

    input_names = config.dataset["input_names"]
    data_type = int if "outcomes" in input_names else bool

    basis = config.dataset["basis"]
    stab_type = f"{basis.lower()}_type"

    dataset_dir = config.experiment_dir / dataset_name
    dataset_params = config.dataset[dataset_name]
    dataset_params["distance"] = config.dataset["distance"]
    noise_params = config.dataset.get("classical_meas_noise_anc")

    datasets = list(
        dataset_generator(dataset_dir, experiment_name, basis, **dataset_params)
    )
    total_samples = sum(len(d.shot) for d in datasets)

    # Convert to desired input
    if noise_params is not None:
        datasets = [add_classical_meas_noise(dataset, noise_params) for dataset in datasets]
    if "repcode" in experiment_name:
        proj_matrix = layout.projection_matrix(stab_type)
        processed = [
            to_inputs(dataset, proj_matrix, input_names=input_names) for dataset in datasets
        ]
    elif "stability" in experiment_name:
        processed = [
            to_inputs_stability(dataset, input_names=input_names) for dataset in datasets
        ]
    else:
        raise ValueError(f"'experiment_name' must contain 'repcode' or 'stability', but {experiment_name} was given.")

    del datasets

    # Process for keras.model input
    inputs = [to_model_input(*arrs, data_type=data_type) for arrs in processed]
    del processed

    return inputs, total_samples
