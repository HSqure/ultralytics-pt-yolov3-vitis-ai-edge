#!/bin/bash

python quant_fast_finetune.py --quant_mode calib --fast_finetune
python quant_fast_finetune.py  --quant_mode test --subset_len 1 --batch_size=1 --fast_finetune --deploy
