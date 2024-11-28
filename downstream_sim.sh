#!/bin/bash

CUDA_VISIBLE_DEVICES="3" nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name lcss > new_train/lcss.log 2>&1 &
# Training with trajectory similarity measure: EDWP
CUDA_VISIBLE_DEVICES="4" nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name edwp > new_train/edwp.log 2>&1 &

# Training with trajectory similarity measure: Discrete Frechet
CUDA_VISIBLE_DEVICES="5" nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name discret_frechet > new_train/discret_frechet.log 2>&1 &

# Training with trajectory similarity measure: EDR
CUDA_VISIBLE_DEVICES="6" nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name edr > new_train/edr.log 2>&1 &

# Training with trajectory similarity measure: Hausdorff
CUDA_VISIBLE_DEVICES="7" nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name hausdorff > new_train/hausdorff.log 2>&1 &

