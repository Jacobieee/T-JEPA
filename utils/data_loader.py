import os
import time
import logging
import pickle5 as pickle
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from config import Config

# 1) read raw pd, 2) split into 3 partitions
def read_traj_dataset(file_path):
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle(file_path)

    l = trajs.shape[0]
    if Config.dataset_prefix == "porto_20200_new" or Config.dataset_prefix == "beijing_new" or Config.dataset_prefix == "geolife":
        train_idx = (int(l*0), Config.train_idx)
    else:
        train_idx = (int(l * 0), int(l * 0.7))
    # train_idx = (int(l*0), 200000)        # for porto
    # train_idx = (int(l * 0), 70000)       # for t-drive
    # train_idx = (int(l * 0), int(l*0.7))  # for tky & nyc
    # train_idx = (int(l * 0), 35000)         # for geolife
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))

    _train = TrajDataset(trajs[train_idx[0]: train_idx[1]])
    _eval = TrajDataset(trajs[eval_idx[0]: eval_idx[1]])
    _test = TrajDataset(trajs[test_idx[0]: test_idx[1]])

    logging.info('[Load traj dataset] END. @={:.0f}, #={}({}/{}/{})' \
                .format(time.time() - _time, l, len(_train), len(_eval), len(_test)))
    return _train, _eval, _test


class TrajDataset(Dataset):
    def __init__(self, data):
        # data: DataFrame
        self.data = data

    def __getitem__(self, index):
        # return traj cells, traj offset, traj time
        traj = np.array(self.data.loc[index].merc_seq)
        # traj_o = np.array(self.data.loc[index].traj_p)
        # traj_t = np.array(self.data.loc[index].time_each_point).reshape(-1, 1)
        # print(traj_o)
        # print(traj_t)
        return traj
        # return traj_o
        # return np.concatenate((traj, traj_o), axis=1)

    def __len__(self):
        return self.data.shape[0]


# Load traj dataset for trajsimi learning
def read_trajsimi_traj_dataset(file_path):
    logging.info('[Load trajsimi traj dataset] START.')
    _time = time.time()

    df_trajs = pd.read_pickle(file_path)
    offset_idx = int(df_trajs.shape[0] * 0.7) # use eval dataset
    df_trajs = df_trajs.iloc[offset_idx : offset_idx + 10000]
    assert df_trajs.shape[0] == 10000
    l = 10000
    # df_trajs = df_trajs.iloc[offset_idx: offset_idx + 5000]
    # assert df_trajs.shape[0] == 5000
    # l = 5000

    train_idx = (int(l*0), int(l*0.7))
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))
    trains = df_trajs.iloc[train_idx[0] : train_idx[1]]
    evals = df_trajs.iloc[eval_idx[0] : eval_idx[1]]
    tests = df_trajs.iloc[test_idx[0] : test_idx[1]]

    logging.info("trajsimi traj dataset sizes. traj: #total={} (trains/evals/tests={}/{}/{})" \
                .format(l, trains.shape[0], evals.shape[0], tests.shape[0]))
    return trains, evals, tests


# Load simi dataset for trajsimi learning
def read_trajsimi_simi_dataset(file_path):
    logging.info('[Load trajsimi simi dataset] START.')
    _time = time.time()
    if not os.path.exists(file_path):
        logging.error('trajsimi simi dataset does not exist')
        exit(200)

    with open(file_path, 'rb') as fh:
        trains_simi, evals_simi, tests_simi, max_distance = pickle.load(fh)
        logging.info("[trajsimi simi dataset loaded] @={}, trains/evals/tests={}/{}/{}" \
                .format(time.time() - _time, len(trains_simi), len(evals_simi), len(tests_simi)))
        return trains_simi, evals_simi, tests_simi, max_distance
