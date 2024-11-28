import sys
sys.path.append('..')
import os
import math
import time
import random
import logging
import torch
import pickle
import pandas as pd
from ast import literal_eval
import numpy as np
from tqdm import tqdm
import traj_dist.distance as tdist
import multiprocessing as mp
from functools import partial
from datetime import datetime
import glob
from datetime import timedelta

from config import Config
from utils import tool_funcs
from utils.cellspace import CellSpace
from utils.tool_funcs import lonlat2meters
from model.node2vec_ import train_node2vec
from utils.edwp import edwp
from utils.data_loader import read_trajsimi_traj_dataset
# from utils.traj import *



def inrange(lon, lat):
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True


def read_plt_file(file_path):
    """Helper function to read plt file and return DataFrame"""
    data = pd.read_csv(file_path, skiprows=6, header=None,
                       names=['Latitude', 'Longitude', 'Unused', 'Altitude', 'Days', 'Date', 'Time'])
    return data


def process_geolife(root_dir):
    labeled_df = pd.DataFrame()
    unlabeled_df = pd.DataFrame()

    # Determine the number of directories for tqdm progress bar
    total_dirs = [os.path.join(root, 'Trajectory') for root, dirs, files in os.walk(root_dir) if 'Trajectory' in dirs]
    pbar = tqdm(total=len(total_dirs), desc='Processing trajectories')

    for root, dirs, files in os.walk(root_dir):
        if 'Trajectory' in dirs:
            trajectory_dir = os.path.join(root, 'Trajectory')
            has_labels = 'labels.txt' in files
            # Read all plt files in the Trajectory directory
            for plt_file in os.listdir(trajectory_dir):
                file_path = os.path.join(trajectory_dir, plt_file)
                if plt_file.endswith('.plt'):
                    data = read_plt_file(file_path)
                    if has_labels:
                        labeled_df = pd.concat([labeled_df, data], ignore_index=True)
                    else:
                        unlabeled_df = pd.concat([unlabeled_df, data], ignore_index=True)
            pbar.update(1)

    pbar.close()

    # Save to CSV
    labeled_df.to_csv('labeled_trajectories.csv', index=False)
    unlabeled_df.to_csv('unlabeled_trajectories.csv', index=False)

    print(labeled_df)
    print(unlabeled_df)

def create_polyline(group):
    trajectory = group[['Longitude', 'Latitude']].apply(
        lambda row: f"[{row['Longitude']}, {row['Latitude']}]", axis=1).tolist()
    trajectory_str = f"[{', '.join(trajectory)}]"
    return trajectory_str


def segment_unlabeled(csv_path):
    data = pd.read_csv(csv_path, parse_dates={'Timestamp': ['Date', 'Time']})

    # Calculate differences between successive timestamps and check if they exceed 10 minutes
    data['Trajectory_ID'] = (data['Timestamp'].diff() > pd.Timedelta(minutes=10)).cumsum()

    # Apply a function to split large groups
    trajectories = []
    for _, group in data.groupby('Trajectory_ID'):
        split_and_add(group, trajectories)

    # Convert list of trajectories into a DataFrame
    trajectories_df = pd.DataFrame(trajectories, columns=['Polyline'])

    # Save DataFrame to CSV
    trajectories_df.to_csv('geolife_unlabeled.csv', index=False)
    print(trajectories_df)
    return trajectories_df

def split_and_add(group, trajectories, max_points=200):
    # Function to split trajectories into segments of at most max_points
    for start in range(0, len(group), max_points):
        end = start + max_points
        # Make sure not to exceed the group's length
        segment = group.iloc[start:min(end, len(group))]
        trajectory = create_polyline(segment)
        trajectories.append({'Polyline': trajectory})


def read_labels(root_dir):
    labels = []
    # Walk through all directories to find labels.txt
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'labels.txt':
                # Read the labels file
                path = os.path.join(subdir, file)
                temp_df = pd.read_csv(path, delimiter='\t')
                # Convert strings to datetime objects for easier manipulation later
                temp_df['Start Time'] = pd.to_datetime(temp_df['Start Time'])
                temp_df['End Time'] = pd.to_datetime(temp_df['End Time'])
                labels.append(temp_df)
    if labels:
        labels_df = pd.concat(labels, ignore_index=True)
        # Filter by transportation modes and correct column name usage
        return labels_df[labels_df['Transportation Mode'].isin(['walk', 'bus', 'bike', 'car'])]
    else:
        return pd.DataFrame()


def process_trajectories(labels_df, trajectories_df_path):
    # Assume 'Timestamp' column in trajectories_df is already in datetime format
    trajectories_df = pd.read_csv(trajectories_df_path, parse_dates={'Timestamp': ['Date', 'Time']})
    final_trajectories = []
    for index, label in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc="Processing labels"):
        # Filtering trajectories that fall within the label's timeframe
        within_range = trajectories_df[
            (trajectories_df['Timestamp'] >= label['Start Time']) &
            (trajectories_df['Timestamp'] <= label['End Time'])
            ]

        # Segment the trajectories if their span is greater than 10 minutes
        if not within_range.empty:
            segments = segment_by_duration(within_range, duration_minutes=10)
            for segment in segments:
                if 20 <= len(segment) <= 100:
                    final_trajectories.append({
                        'Start Time': label['Start Time'],
                        'End Time': label['End Time'],
                        'Mode': label['Transportation Mode'],
                        'Trajectory': create_polyline(segment)
                    })
                elif len(segment) > 100:
                    sampled_segment = segment.sample(n=100)
                    final_trajectories.append({
                        'Start Time': label['Start Time'],
                        'End Time': label['End Time'],
                        'Mode': label['Transportation Mode'],
                        'Trajectory': create_polyline(sampled_segment)
                    })

    final = pd.DataFrame(final_trajectories)
    final.to_csv("geolife_labeled.csv", index=False)
    mode_counts = final['Mode'].value_counts()
    print("Number of trajectories per transportation mode:")
    for mode, count in mode_counts.items():
        print(f"{mode}: {count}")


def segment_by_duration(df, duration_minutes=10):
    segments = []
    start_index = 0  # Use an index rather than the DataFrame row itself

    # Iterate over the DataFrame rows using index
    for i in range(1, len(df)):
        # Check if the time difference exceeds the threshold
        if (df.iloc[i]['Timestamp'] - df.iloc[start_index]['Timestamp']).total_seconds() / 60 > duration_minutes:
            # Append the segment from start_index to i (not inclusive of i)
            segments.append(df.iloc[start_index:i])
            start_index = i  # Update start_index to current position

    # Append the last segment from the last start_index to the end of the DataFrame
    segments.append(df.iloc[start_index:])  # This includes the last segment

    return segments

# def process_geolife(directory):
#     """
#     Count the number of .plt files and labels.txt files in a given directory and its subdirectories.
#
#     Parameters:
#     - directory (str): Path to the directory to search.
#
#     Returns:
#     - tuple: (number of .plt files, number of labels.txt files)
#     """
#     total_plt_count = 0
#     labels_count = 0
#     plt_in_labels_dir_count = 0
#
#     # Walk through all directories and files in the root directory
#     for root, dirs, files in os.walk(directory):
#         # Count total .plt files
#         for file in files:
#             if file.endswith('.plt'):
#                 total_plt_count += 1
#
#         # Check if labels.txt is in the current directory
#         if 'labels.txt' in files:
#             labels_count += 1
#             # Path to the 'Trajectory' subdirectory
#             trajectory_dir = os.path.join(root, 'Trajectory')
#             # Count the .plt files in the 'Trajectory' directory
#             if os.path.exists(trajectory_dir):
#                 for traj_file in os.listdir(trajectory_dir):
#                     if traj_file.endswith('.plt'):
#                         plt_in_labels_dir_count += 1
#
#     return total_plt_count, labels_count, plt_in_labels_dir_count


def clean_and_output_data():
    _time = time.time()
    # cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
    # embs = pickle.load(open(Config.dataset_embs_file, 'rb'))
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00339/
    # download train.csv.zip and unzip it. rename train.csv to porto.csv
    dfraw = pd.read_csv(Config.root_dir + '/data/geolife.csv')
    # print(dfraw)
    dfraw = dfraw.rename(columns = {"Polyline": "wgs_seq"})
    # print(dfraw)
    # dfraw = dfraw[dfraw.MISSING_DATA == False]
    # print(dfraw.groupby("TAXI_ID").size())
    # print(dfraw["TIMESTAMP"].loc[0])
    # length requirement
    # print(dfraw.head())

    """
    """

    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
    logging.info('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

    # all_coords = np.concatenate(dfraw['wgs_seq'].values)
    # # print(all_coords)
    # # Calculate the overall MBR
    # overall_mbr = calculate_overall_mbr(all_coords)
    # print(overall_mbr)



    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj) ) # True: valid
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))
    total_points = dfraw['trajlen'].sum()
    print(f"num of total points: {total_points}.")
    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])


    # dfraw['traj_p'] = dfraw['merc_seq'].apply(lambda traj: convert_trajectory(traj, cellspace))

    # def expand_timestamps(row):
    #     """
    #     This function takes a row of the DataFrame,
    #     extracts the sequence and the starting timestamp,
    #     and returns a list of timestamps increased by 15 seconds for each element in the sequence.
    #     """
    #     sequence = row['merc_seq']
    #     start_time = row['TIMESTAMP']
    #     return [start_time + i * 15 for i in range(len(sequence))]

    # dfraw['time_each_point'] = dfraw.apply(expand_timestamps, axis=1)
    # print(dfraw['time_each_point'])

    logging.info('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq']].reset_index(drop = True)
    print(dfraw.columns)
    dfraw.to_pickle(Config.dataset_file)
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    return


def init_cellspace():
    # 1. create cellspase
    # 2. initialize cell embeddings (create graph, train, and dump to file)

    x_min, y_min = lonlat2meters(Config.min_lon, Config.min_lat)
    x_max, y_max = lonlat2meters(Config.max_lon, Config.max_lat)
    x_min -= Config.cellspace_buffer
    y_min -= Config.cellspace_buffer
    x_max += Config.cellspace_buffer
    y_max += Config.cellspace_buffer

    cell_size = int(Config.cell_size)
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    with open(Config.dataset_cell_file, 'wb') as fh:
        pickle.dump(cs, fh, protocol = pickle.HIGHEST_PROTOCOL)

    _, edge_index = cs.all_neighbour_cell_pairs_permutated_optmized()
    edge_index = torch.tensor(edge_index, dtype = torch.long, device = Config.device).T
    train_node2vec(edge_index)
    return


def generate_newsimi_test_dataset():
    trajs = pd.read_pickle(Config.dataset_file)  # using test part only
    # print(Config.dataset_file)
    l = trajs.shape[0]
    print(l)
    n_query = 1000
    n_db = 10000
    test_idx = (int(l * 0.8), int(l * 0.8) + n_db)
    test_trajs = trajs[test_idx[0]: test_idx[1]]
    logging.info("Test trajs loaded.")

    # for varying db size
    def _raw_dataset():
        query_lst = []  # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.iteritems():
            if i < n_query:
                query_lst.append(np.array(v)[::2].tolist())
            db_lst.append(np.array(v)[1::2].tolist())
            i += 1

        output_file_name = Config.dataset_file + '_newsimi_raw.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_raw_dataset done.")
        return

    # for varying downsampling rate
    def _downsample_dataset(rate):
        unrate = 1 - rate  # preserved rate
        query_lst = []  # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.iteritems():
            if i < n_query:
                _q = np.array(v)[::2]
                _q_len = _q.shape[0]
                _idx = np.sort(np.random.choice(_q_len, math.ceil(_q_len * unrate), replace=False))
                query_lst.append(_q[_idx].tolist())
            _db = np.array(v)[1::2]
            _db_len = _db.shape[0]
            _idx = np.sort(np.random.choice(_db_len, math.ceil(_db_len * unrate), replace=False))
            db_lst.append(_db[_idx].tolist())
            i += 1

        output_file_name = Config.dataset_file + '_newsimi_downsampling_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_downsample_dataset done. rate={}".format(rate))
        return

    # for varying distort rate
    def _distort_dataset(rate):
        query_lst = []  # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.iteritems():
            if i < n_query:
                _q = np.array(v)[::2]
                for _row in range(_q.shape[0]):
                    if random.random() < rate:
                        _q[_row] = _q[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
                query_lst.append(_q.tolist())

            _db = np.array(v)[1::2]
            for _row in range(_db.shape[0]):
                if random.random() < rate:
                    _db[_row] = _db[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
            db_lst.append(_db.tolist())
            i += 1

        output_file_name = Config.dataset_file + '_newsimi_distort_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_distort_dataset done. rate={}".format(rate))
        return

    _raw_dataset()

    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _downsample_dataset(rate)

    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _distort_dataset(rate)

    return


# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='hausdorff'):
    # 1. read trajs from file, and split to 3 datasets, and data normalization
    # 2. calculate simi in 3 datasets separately.
    # 3. dump 3 datasets into files

    logging.info("traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()

    # 1.
    trains, evals, tests = read_trajsimi_traj_dataset(Config.dataset_file)
    trains, evals, tests = _normalization([trains, evals, tests])

    logging.info("traj dataset sizes. traj: trains/evals/tests={}/{}/{}" \
                 .format(trains.shape[0], evals.shape[0], tests.shape[0]))

    # 2.
    fn = _get_simi_fn(fn_name)
    tests_simi = _simi_matrix(fn, tests)
    evals_simi = _simi_matrix(fn, evals)
    trains_simi = _simi_matrix(fn, trains)  # [ [simi, simi, ... ], ... ]

    max_distance = max(max(map(max, trains_simi)), max(map(max, evals_simi)), max(map(max, tests_simi)))

    _output_file = '{}_traj_simi_dict_{}.pkl'.format(Config.dataset_file, fn_name)
    with open(_output_file, 'wb') as fh:
        tup = trains_simi, evals_simi, tests_simi, max_distance
        pickle.dump(tup, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("traj_simi_computation ends. @={:.3f}".format(time.time() - _time))
    return tup


def _normalization(lst_df):
    # lst_df: [df, df, df]
    xs = []
    ys = []
    for df in lst_df:
        for _, v in df.merc_seq.iteritems():
            arr = np.array(v)
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    mean = np.array([xs.mean(), ys.mean()])
    std = np.array([xs.std(), ys.std()])

    for i in range(len(lst_df)):
        lst_df[i].merc_seq = lst_df[i].merc_seq.apply(lambda lst: ((np.array(lst) - mean) / std).tolist())

    return lst_df


def _get_simi_fn(fn_name):
    fn = {'lcss': tdist.lcss, 'edr': tdist.edr, 'frechet': tdist.frechet,
          'discret_frechet': tdist.discret_frechet,
          'hausdorff': tdist.hausdorff, 'edwp': edwp}.get(fn_name, None)
    if fn_name == 'lcss' or fn_name == 'edr':
        fn = partial(fn, eps=Config.test_exp1_lcss_edr_epsilon)
    return fn


def _simi_matrix(fn, df):
    _time = time.time()

    l = df.shape[0]
    batch_size = 50
    assert l % batch_size == 0

    # parallel init
    tasks = []
    for i in range(math.ceil(l / batch_size)):
        if i < math.ceil(l / batch_size) - 1:
            tasks.append((fn, df, list(range(batch_size * i, batch_size * (i + 1)))))
        else:
            tasks.append((fn, df, list(range(batch_size * i, l))))

    num_cores = int(mp.cpu_count()) - 6
    assert num_cores > 0
    logging.info("pool.size={}".format(num_cores))
    pool = mp.Pool(num_cores)
    lst_simi = pool.starmap(_simi_comp_operator, tasks)
    pool.close()

    # extend lst_simi to matrix simi and pad 0s
    lst_simi = sum(lst_simi, [])
    for i, row_simi in enumerate(lst_simi):
        lst_simi[i] = [0] * (i + 1) + row_simi
    assert sum(map(len, lst_simi)) == l ** 2
    logging.info('simi_matrix computation done., @={}, #={}'.format(time.time() - _time, len(lst_simi)))

    return lst_simi


# async operator
def _simi_comp_operator(fn, df_trajs, sub_idx):
    simi = []
    l = df_trajs.shape[0]
    for _i in sub_idx:
        t_i = np.array(df_trajs.iloc[_i].merc_seq)
        simi_row = []
        for _j in range(_i + 1, l):
            t_j = np.array(df_trajs.iloc[_j].merc_seq)
            val = fn(t_i, t_j)
            if val is not None:
                simi_row.append(float(val))
        simi.append(simi_row)
    logging.debug('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
                  .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi

def read_ft():
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle("../data/geolife")

    l = trajs.shape[0]
    print(l)
    # train_idx = (int(l*0), 200000)        # for porto
    # train_idx = (int(l * 0), 70000)  # for t-drive
    # train_idx = (int(l * 0), int(l*0.7))
    # train_idx = (int(l * 0), 35000)         # for geolife
    offset_idx = int(trajs.shape[0] * 0.7)  # use eval dataset
    print(offset_idx)
    df_trajs = trajs.iloc[offset_idx: offset_idx + 5000]
    # print(df_trajs)
    # print(df_trajs["wgs_seq"])
    def array_to_polyline(coordinates):
        # Create the trajectory string for the entire array
        trajectory = [f"[{coord[0]}, {coord[1]}]" for coord in coordinates]
        trajectory_str = f"[{', '.join(trajectory)}]"
        return trajectory_str

    df_trajs['POLYLINE'] = df_trajs["wgs_seq"].apply(array_to_polyline)
    print(df_trajs['POLYLINE'])
    print(df_trajs.shape)
    df_trajs.to_csv("geolife_ft.csv", index=False)


def clean_label():
    _time = time.time()

    # root_dir = os.path.dirname(__file__)
    csv_input_path = '../data/geolife_labeled.csv'
    # dataset_file = root_dir + '../data/geolife_labeled.pkl'

    dfraw = pd.read_csv(csv_input_path)
    dfraw = dfraw.rename(columns={"Trajectory": "wgs_seq"})

    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
    logging.info('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(
        lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))  # True: valid
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))

    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])
    logging.info('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))

    dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq', 'Mode']].reset_index(drop=True)

    # dfraw.to_pickle(dataset_file)
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    return

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'),
                                    logging.StreamHandler()]
                        )
    Config.dataset = 'geolife'
    Config.post_value_updates()

    # init_cellspace()
    # clean_and_output_data()
    # generate_newsimi_test_dataset()
    # traj_simi_computation("hausdorff")
    # traj_simi_computation("discret_frechet")
    # traj_simi_computation("edr")
    # traj_simi_computation("lcss")
    # traj_simi_computation("edwp")
    # process_geolife('../../Geolife Trajectories 1.3')
    # segment_unlabeled('unlabeled_trajectories.csv')
    # raw_labeled = read_labels('../../Geolife Trajectories 1.3')
    # print(raw_labeled)
    # process_trajectories(raw_labeled, "labeled_trajectories.csv")
    # read_ft()
    # clean_label()

