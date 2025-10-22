import os
import pathlib
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from qrennd import (
    Layout,
    Config,
)

from lib.util import load_datasets
from lib.callbacks import get_callbacks
from lib.sequences import Sequence_stability
from lib.models import get_model

# better error descriptions
keras.config.disable_traceback_filtering()

######################################

# Parameters
LAYOUT_FILE = "stability_a4_bZ.yaml"
CONFIG_FILE = "config_nslru_stab.yaml"

PARENT_DIR = pathlib.Path.cwd()
DATA_DIR = PARENT_DIR / "data"
OUTPUT_DIR = PARENT_DIR / "output"
CONFIG_DIR = PARENT_DIR / "configs"

###################################

# Load setup objects
config = Config.from_yaml(
    filepath=CONFIG_DIR / CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)
config.log_dir.mkdir(exist_ok=True, parents=True)
config.checkpoint_dir.mkdir(exist_ok=True, parents=True)

LAYOUT_DIR = DATA_DIR / config.experiment / "config"
layout = Layout.from_yaml(LAYOUT_DIR / LAYOUT_FILE)

# set random seed for tensorflow, numpy and python
# ensure that the new seed is stored in config for reproducible results
if config.seed is None:
    config.seed = int(np.random.rand() * (2**32 - 1))
random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

config.to_yaml(config.run_dir / "config.yaml")

# load datasets
val_data, val_shots = load_datasets(config=config, layout=layout, dataset_name="val")
train_data, train_shots = load_datasets(
    config=config, layout=layout, dataset_name="train"
)

# this is for model.fit to know that the num_rounds coordinate is not fixed
batch_size = config.train["batch_size"]
tensor1, tensor2 = train_data[0], train_data[-1]
seq1, seq2 = Sequence_stability(*tensor1, batch_size), Sequence_stability(*tensor2, batch_size)
first_batch, second_batch = seq1[0], seq2[0]


def infinite_gen(inputs):
    sequences = [Sequence_stability(*tensors, batch_size) for tensors in inputs]
    indices = []
    for k, sequence in enumerate(sequences):
        inds = [(k, i) for i in range(sequence._num_batches)]
        indices += inds

    while True:
        # this is for model.fit to know that the num_rounds coordinate is not fixed
        yield first_batch
        yield second_batch

        random.shuffle(indices)

        for k, ind in indices:
            yield sequences[k][ind]


# load model
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
callbacks = get_callbacks(config)


# train model
train = config.dataset["train"]
val = config.dataset["val"]
batch_size = config.train["batch_size"]
history = model.fit(
    infinite_gen(train_data),
    validation_data=infinite_gen(val_data),
    epochs=config.train["epochs"],
    callbacks=callbacks,
    verbose=0,
    steps_per_epoch=train_shots // batch_size
    + 2,  # +2 is for model.fit to know that the num_rounds coordinate is not fixed
    validation_steps=val_shots // batch_size
    + 2,  # +2 is for model.fit to know that the num_rounds coordinate is not fixed
)
model.save(config.checkpoint_dir / "final_weights.keras")
