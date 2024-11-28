#!/bin/bash

CUDA_VISIBLE_DEVICES="4" nohup python train_trajsimi.py --dataset geolife --trajsimi_measure_fn_name lcss > new_train/geolife_freeze_lcss.log 2>&1 &
# Training with trajectory similarity measure: EDWP
#CUDA_VISIBLE_DEVICES="4" nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name edwp > new_train/freeze_edwp.log 2>&1 &

# Training with trajectory similarity measure: Discrete Frechet
CUDA_VISIBLE_DEVICES="5" nohup python train_trajsimi.py --dataset geolife --trajsimi_measure_fn_name discret_frechet > new_train/geolife_freeze_discret_frechet.log 2>&1 &

# Training with trajectory similarity measure: EDR
CUDA_VISIBLE_DEVICES="6" nohup python train_trajsimi.py --dataset geolife --trajsimi_measure_fn_name edr > new_train/geolife_freeze_edr.log 2>&1 &

# Training with trajectory similarity measure: Hausdorff
CUDA_VISIBLE_DEVICES="7" nohup python train_trajsimi.py --dataset geolife --trajsimi_measure_fn_name hausdorff > new_train/geolife_freeze_hausdorff.log 2>&1 &