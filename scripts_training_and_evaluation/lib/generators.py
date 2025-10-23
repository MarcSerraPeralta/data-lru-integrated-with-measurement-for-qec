from typing import Generator, List

import xarray as xr


def dataset_generator(
    datasets_dir: str,
    experiment_name: str,
    basis: str,
    states: List[str],
    rounds: List[int],
    shots: int = None,
    **args,
) -> Generator:
    for num_rounds in rounds:
        for state in states:
            experiment = experiment_name.format(
                basis=basis,
                state=state,
                shots=shots,
                num_rounds=num_rounds,
                **args,
            )
            try:
                dataset = xr.load_dataset(datasets_dir / experiment / "measurements.nc")
            except FileNotFoundError as error:
                raise ValueError(
                    f"Invalid experiment data directory: {datasets_dir / experiment / 'measurements.nc'}"
                ) from error

            yield dataset
