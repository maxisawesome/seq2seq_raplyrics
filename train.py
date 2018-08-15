import argparse
import json
import importlib
from typing import Dict
import os
from data.base import LyricGenerator
from network.encoder import encoderRNN
from network.decoder import decoderRNN
from util import train_model

def run_experiment(experiment_config: Dict, save_weights: bool, use_wandb: bool=False):
    """
    experiment_config is of the form
    {
        "dataset_args": {
            "max_len": 20
        },
        "network_args": {
            "hidden": 1024,
            "phoneme_embedding": 128
        },
        "train_args": {
            "batch_size": 16,
            "epochs": 10
        }
    }
    save_weights: if True, will save the final model weights to a canonical location (see Model in models/base.py)
    """
    #print(f'Running experiment with config {experiment_config}') # on GPU {gpu_ind}')
    #datasets_module = importlib.import_module('seq2seq_raplyrics.data')
    #dataset_class_ = getattr(datasets_module, 'LyricGenerator')
    data = LyricGenerator(experiment_config["dataset_args"]["max_len"])
    data.load_data()
    encoder = encoderRNN(data.n_phonemes, experiment_config["network_args"]["phoneme_embedding"], experiment_config["network_args"]["hidden"])
    decoder = decoderRNN(data.n_phonemes, data.n_words, experiment_config["network_args"]["phoneme_embedding"], experiment_config["network_args"]["hidden"], experiment_config["dataset_args"]["max_len"])
    print(encoder)
    print(decoder)
    print("Total sentences: %d" % (len(data),))

    train_model(
        data,
        encoder,
        decoder,
        experiment_config["train_args"]["batch_size"],
        experiment_config["train_args"]["epochs"]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final weights will be saved to canonical, version-controlled location"
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="JSON of experiment to run (e.g. '{\"dataset_args\": {\"max_len\": 20},\"network_args\": {\"hidden\": 1024,\"phoneme_embedding\": 128},\"train_args\": {\"batch_size\": 16,\"epochs\": 10}}"
    )

    args = parser.parse_args()
    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config, args.save)

