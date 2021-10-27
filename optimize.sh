#!/bin/bash

export XILINXD_LICENSE_FILE=/workspace/alinx/Xilinx.lic

python optimizer.py --quant_mode calib
python optimizer.py --quant_mode test --subset_len 1 --batch_size=1 --deploy
