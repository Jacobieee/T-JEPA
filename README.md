# T-JEPA

This is the official code for [T-JEPA: A Joint-Embedding Predictive Architecture for Trajectory Similarity Computation](https://arxiv.org/abs/2406.12913). This paper has been accepted by SIGSPATIAL 2024.

If you find this work useful for your research, please cite:
```
@inproceedings{li2024t,
  title={T-JEPA: A Joint-Embedding Predictive Architecture for Trajectory Similarity Computation},
  author={Li, Lihuan and Xue, Hao and Song, Yang and Salim, Flora},
  booktitle={Proceedings of the 32nd ACM International Conference on Advances in Geographic Information Systems},
  pages={569--572},
  year={2024}
}
```

## Run T-JEPA
To pre-train T-JEPA:
```
python train.py --dataset porto
```
And to run downstream fine-tuning (approximate heuristic measures) after pre-training:
```
python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name hausdorff
```
We follow the preprocessing protocol of [TrajCL](https://github.com/changyanchuan/TrajCL). Please refer to TrajCL for data preparation.



## Acknowledgement
The code is largely borrowed from [TrajCL](https://github.com/changyanchuan/TrajCL).
