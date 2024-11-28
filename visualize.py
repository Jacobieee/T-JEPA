# import math
import logging
# import random
import time
import pandas as pd
import pickle5 as pickle
# import pickle
# import torch

# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
# from torch.nn.utils.rnn import pad_sequence
#
# from config import Config as Config
# from utils import tool_funcs
# # from utils.data_loader import read_trajsimi_simi_dataset, read_trajsimi_traj_dataset
# from utils.traj import merc2cell2, generate_spatial_features, merc2cell
# from model.graph_func import *
# from utils.preprocessing_porto import get_offset


# def load_trajsimi_dataset():
#     # read (1) traj dataset for trajsimi, (2) simi matrix dataset for trajsimi
#     trajsimi_traj_dataset_file = Config.dataset_file
#     trajsimi_simi_dataset_file = '{}_traj_simi_dict_{}.pkl'.format( \
#         Config.dataset_file, Config.trajsimi_measure_fn_name)
#
#     trains_traj, evals_traj, tests_traj = read_trajsimi_traj_dataset(trajsimi_traj_dataset_file)
#     trains_traj, evals_traj, tests_traj = trains_traj.merc_seq.values, evals_traj.merc_seq.values, tests_traj.merc_seq.values
#     trains_simi, evals_simi, tests_simi, max_distance = read_trajsimi_simi_dataset(trajsimi_simi_dataset_file)
#
#     # trains_traj : [[[lon, lat_in_merc], [], ..], [], ...]
#     # trains_simi : list of list
#     return {'trains_traj': trains_traj, 'evals_traj': evals_traj, 'tests_traj': tests_traj, \
#             'trains_simi': trains_simi, 'evals_simi': evals_simi, 'tests_simi': tests_simi, \
#             'max_distance': max_distance}

def read_trajsimi_traj_dataset(file_path):
    logging.info('[Load trajsimi traj dataset] START.')
    _time = time.time()

    # df_trajs = pd.read_pickle(file_path)
    df_trajs = pickle.load(file_path)
    print(df_trajs)
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



def load_trajsimi_dataset():
    # read (1) traj dataset for trajsimi, (2) simi matrix dataset for trajsimi
    trajsimi_traj_dataset_file = "./porto_20200_new_traj_simi_dict_hausdorff.pkl"
    _, _, tests_traj = read_trajsimi_traj_dataset(trajsimi_traj_dataset_file)
    tests_traj = tests_traj.wgs_seq.values
    return tests_traj
    #
    # trains_traj, evals_traj, tests_traj = read_trajsimi_traj_dataset(trajsimi_traj_dataset_file)
    # trains_traj, evals_traj, tests_traj = trains_traj.merc_seq.values, evals_traj.merc_seq.values, tests_traj.merc_seq.values
    # trains_simi, evals_simi, tests_simi, max_distance = read_trajsimi_simi_dataset(trajsimi_simi_dataset_file)

    # trains_traj : [[[lon, lat_in_merc], [], ..], [], ...]
    # trains_simi : list of list
    # return {'trains_traj': trains_traj, 'evals_traj': evals_traj, 'tests_traj': tests_traj, \
    #         'trains_simi': trains_simi, 'evals_simi': evals_simi, 'tests_simi': tests_simi, \
    #         'max_distance': max_distance}



if __name__ == '__main__':
    # logging.basicConfig(level = logging.DEBUG,
    #                     format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
    #                     handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'),
    #                                 logging.StreamHandler()]
    #                     )
    # Config.dataset = 'porto'
    # Config.post_value_updates()

    porto_matrix = np.array([
        [ 648, 1781,  201, 1293,  475,  149],
        [ 300,  891,   45,   59,  907, 1012],
        [ 334, 1856,  734,  508,  589, 1110],
        [1194, 1964, 1196,  278, 1765, 1126],
        [ 435,   54, 1367,  812,  559,  919],
        [ 129, 1521, 1027, 1370,  277, 1358],
        [1964, 1194, 1196,  278, 1765, 1126],
        [ 394,   90,  596,  544,  939, 1804],
        [1837,  746, 1111,  788,  984, 1020],
        [ 185, 1025, 1649,  432,  105, 1822]
    ])
    traj_db = load_trajsimi_dataset()

    for trajs in porto_matrix:
        # idx = trajs[1:]
        idx = trajs[1]
        # print(traj_db[idx])
        traj = traj_db[idx]
        center_lat = sum(point[1] for point in traj) / len(traj)
        center_lon = sum(point[0] for point in traj) / len(traj)