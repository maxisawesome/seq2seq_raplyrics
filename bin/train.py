import argparse
import json
import importlib
import os

def run_experiement(experiment_config: Dict, save_weights: bool, use_wandb: bool=False):
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
            "batch_size": 128,
            "epochs": 10,
        }
    }
    save_weights: if True, will save the final model weights to a canonical location (see Model in models/base.py)
    """
    print(f'Running experiment with config {experiment_config}') # on GPU {gpu_ind}')
    datasets_module = importlib.import_module('seq2seq_raplyrics.datasets')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden",
        type=int,
        default=1024,
        help="Number of hidden layers to use in the model"
    )
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final weights will be saved to canonical, version-controlled location"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default="35",
        help="Maximum length of the encoder sentences"
    )
    parser.add_argument(
        "--tfr",
        type=float,
        default=.2,
        help="fraction of examples to use teacher forcing on"
    )
    args = parser.parse_args()
    print(args)
    #run_experiment(experiment_config, args.save, args.gpu)

