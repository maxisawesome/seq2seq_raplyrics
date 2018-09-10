import argparse
import json
import importlib
from typing import Dict
import os
import torch
from torch.utils.data import Dataset, DataLoader
from data.base import LyricGenerator
from network.encoder import encoderRNN
from network.decoder import decoderRNN
from util import train_model, show_batch

def run_experiment(experiment_config: Dict, save_weights: bool, use_wandb: bool=False):
    """
    experiment_config is of the form
    {
        "max_len": 20,
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
    print(experiment_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = LyricGenerator(experiment_config["max_len"])
    data.load_data()
    dataloader = DataLoader(data, batch_size=experiment_config["train_args"]['batch_size'], shuffle=True)
    #show_batch(data, dataloader)
    encoder = encoderRNN(data.n_phonemes, experiment_config["network_args"]["phoneme_embedding"], experiment_config["network_args"]["hidden"]).to(device)
    decoder = decoderRNN(data.n_phonemes, experiment_config["network_args"]["phoneme_embedding"], experiment_config["network_args"]["hidden"], experiment_config["max_len"]).to(device)
    print(encoder)
    print(decoder)
    print("Total sentences: %d" % (len(data),))

    train_model(
        dataloader,
        encoder,
        decoder,
        experiment_config["train_args"]["batch_size"],
        experiment_config["train_args"]["epochs"],
        .3
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

