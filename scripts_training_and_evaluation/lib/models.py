from tensorflow import keras

from qrennd.configs import Config
from qrennd.models.networks import (
    eval_network,
    lstm_network,
)
from qrennd.models.models import (
    lstm_model,
    convlstm_model,
    conv_lstm_model,
    lstm_decoder_model,
)

DEFAULT_OPT_PARAMS = dict(learning_rate=0.001)


def lstm_stability_model(
    rec_features: int, eval_features: int, config: Config
) -> keras.Model:
    rec_input = keras.layers.Input(
        shape=(None, rec_features),
        dtype="float32",
        name="rec_input",
    )
    eval_input = keras.layers.Input(
        shape=(eval_features,),
        dtype="float32",
        name="eval_input",
    )

    if "main_eval_init" in config.model:
        main_eval_params = config.model["main_eval_init"]
        network = eval_network(name="i", **main_eval_params)
        main_output_init = next(network)(eval_input)
        for layer in network:
            main_output_init = layer(main_output_init)
    else:
        main_output_init = eval_input

    reshape_layer = keras.layers.Reshape((-1, rec_features), name="iresh")
    main_output_init_reshaped = reshape_layer(main_output_init)
    concat_layer = keras.layers.Concatenate(axis=1, name="iconc")
    main_input = concat_layer((main_output_init_reshaped, rec_input))

    lstm_params = config.model["LSTM"]
    network = lstm_network(name="LSTM", **lstm_params)
    output = next(network)(main_input)
    for layer in network:
        output = layer(output)

    activation_layer = keras.layers.Activation(
        activation="relu",
        name="relu_LSTM",
    )
    output = activation_layer(output)

    main_eval_params = config.model["main_eval_final"]
    network = eval_network(name="main", **main_eval_params)
    main_output = next(network)(output)
    for layer in network:
        main_output = layer(main_output)

    inputs = [rec_input, eval_input]
    outputs = [main_output]

    model = keras.Model(inputs=inputs, outputs=outputs, name="stabm")

    opt_params = config.train.get("optimizer", DEFAULT_OPT_PARAMS)
    optimizer = keras.optimizers.Adam(**opt_params)

    loss = config.train.get("loss")
    loss_weights = config.train.get("loss_weights")
    metrics = config.train.get("metrics")

    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights
    )

    if config.init_weights:
        try:
            experiment_dir = config.output_dir / config.experiment
            model.load_weights(experiment_dir / config.init_weights)
        except FileNotFoundError as error:
            raise ValueError(
                "Invalid initial weights in configuration file."
            ) from error

    return model


def eval_model(rec_features: int, eval_features: int, config: Config) -> keras.Model:
    rec_input = keras.layers.Input(
        shape=(None, rec_features),
        dtype="float32",
        name="rec_input",
    )
    eval_input = keras.layers.Input(
        shape=(eval_features,),
        dtype="float32",
        name="eval_input",
    )

    main_eval_params = config.model["main_eval"]
    network = eval_network(name="main", **main_eval_params)
    main_output = next(network)(eval_input)
    for layer in network:
        main_output = layer(main_output)

    # mimic structure of other models
    network = eval_network(name="aux", units=[1])
    aux_output = next(network)(eval_input)
    for layer in network:
        aux_output = layer(aux_output)

    inputs = [rec_input, eval_input]
    outputs = [main_output, aux_output]

    model = keras.Model(inputs=inputs, outputs=outputs, name="eval_model")

    opt_params = config.train.get("optimizer", DEFAULT_OPT_PARAMS)
    optimizer = keras.optimizers.Adam(**opt_params)

    loss = config.train.get("loss")
    loss_weights = config.train.get("loss_weights")
    metrics = config.train.get("metrics")

    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights
    )

    if config.init_weights:
        try:
            experiment_dir = config.output_dir / config.experiment
            model.load_weights(experiment_dir / config.init_weights)
        except FileNotFoundError as error:
            raise ValueError(
                "Invalid initial weights in configuration file."
            ) from error

    return model


def get_model(
    rec_features: int,
    eval_features: int,
    config: Config,
) -> keras.Model:
    model_type = config.model["type"]
    if model_type == "LSTM":
        return lstm_model(rec_features, eval_features, config)
    if model_type == "LSTM-stability":
        return lstm_stability_model(rec_features, eval_features, config)
    elif model_type == "ConvLSTM":
        return convlstm_model(rec_features, eval_features, config)
    elif model_type == "Conv_LSTM":
        return conv_lstm_model(rec_features, eval_features, config)
    elif model_type == "LSTM_decoder":
        return lstm_decoder_model(rec_features, eval_features, config)
    elif model_type == "eval":
        return eval_model(rec_features, eval_features, config)
    else:
        raise ValueError(f"Unrecognize config.model.type {model_type}")

