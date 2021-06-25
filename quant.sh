#!/bin/bash

python quant.py --quant_mode calib
python quant.py --quant_mode test --subset_len 1 --batch_size=1 --deploy
