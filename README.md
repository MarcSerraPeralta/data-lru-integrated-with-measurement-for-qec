# data-lru-integrated-with-measurement-for-qec

Scripts to replicate the results from the project _arXiv:2511.17460 (2025)_ 
about LRU integrated with measurement for QEC.


## Steps

_Note: the commands are given for UNIX-like systems_

Clone this repository:
```
git clone git@github.com:MarcSerraPeralta/data-lru-integrated-with-measurement-for-qec.git
cd data-lru-integrated-with-measurement-for-qec
```
Download the experimental data and decompress it:
```
cd scripts_data_pre-processing/
curl https://data.4tu.nl/file/902d7f9e-38bf-48a2-a2f7-52bbf7aeeedf/c2e0c7c1-60b7-493f-baeb-1791a6e85cc5 --output data.zip
unzip data.zip -d data_external
rm data.zip
```
Edit the lines `DC_DATA_DIR = pathlib.Path("...")` in the scripts inside `scripts_data_pre-processing`
to point to the specific experiment subdirectory (inside `data_external/` to pre-process.

Set up python and run the scripts:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python dclab-format_to_qrennd-format_memory-experiment_experimental-data.py
```
Move the data to the training directory:
```
mv data ../scripts_training_and_evaluation/data
cd ../scripts_training_and_evaluation
```
Specify the neural-network hyperparameters in a config YAML file inside `config/`
(e.g. copy one of the `config.yaml` files from the directories in `model_weights/`).
Specify the config file in the training scripts (`train_memory.py` or `train_stability.py`).
Set up python (requires version 3.11.7 due to tensorflow) and run the scripts:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements_train_python-3-11-7.txt

python train_memory.py
```
Specify the experiment directory in the evaluation scripts (`evaluate_memory.py` or `evaluate_stability.py`).
```
python evaluate_memory_experiment.py
```
Move the trained neural-networks to the ensembling directory:
```
mv output ../scripts_ensembling/output
cd ../scripts_ensembling
```
Specify the ensembling to use in `evaluate_ensembles.py` (see TXT files inside `ensembles/`).
```
python evaluate_ensembles.py
```


## Model weights

The model weights of the neural networks used to obtain the results in _arXiv:2511.17460 (2025)_
can be found in `model_weights/`.


## Figure data

The data used in the Jupyter notebook of _https://doi.org/10.4121/902d7f9e-38bf-48a2-a2f7-52bbf7aeeedf_
to plot the figures in _arXiv:2511.17460 (2025)_ can be found in `scripts_figures/`.
