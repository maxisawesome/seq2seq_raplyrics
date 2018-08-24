#!/bin/sh
python train.py --save '{ "max_len": 15, "network_args": { "hidden":1024, "phoneme_embedding":128 }, "train_args": {"batch_size": 16, "epochs": 10 } }'
