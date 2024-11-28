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
import traj_dist.distance as tdist
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from config import Config
# from utils import tool_funcs
from utils.cellspace import CellSpace
# from utils.tool_funcs import lonlat2meters
# from model.node2vec_ import train_node2vec
from utils.edwp import edwp
from utils.data_loader import read_trajsimi_traj_dataset
# from utils.traj import *

def inrange(lon, lat):
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True


def get_offset(x, y , cellspace):
    """Convert a single point to cell ID and offset."""
    assert cellspace.x_min <= x <= cellspace.x_max and cellspace.y_min <= y <= cellspace.y_max
    cell_id = cellspace.get_cellid_by_point(x, y)
    i_x, i_y = cellspace.get_xyidx_by_cellid(cell_id)
    cell_x_min = cellspace.x_min + i_x * cellspace.x_unit
    cell_y_min = cellspace.y_min + i_y * cellspace.y_unit
    offset_x = x - cell_x_min
    offset_y = y - cell_y_min
    assert 0 <= offset_x < 100 and 0 <= offset_y < 100
    return (i_x, i_y, offset_x, offset_y)

def inv_offset(cell_id, offset_x, offset_y, cellspace):
    # Convert cell ID back to grid indices
    i_x, i_y = cellspace.get_xyidx_by_cellid(cell_id)

    # Calculate the bottom-left corner of the cell
    cell_x_min = cellspace.x_min + i_x * cellspace.x_unit
    cell_y_min = cellspace.y_min + i_y * cellspace.y_unit

    # Reconstruct the original Mercator coordinates
    original_x = cell_x_min + offset_x
    original_y = cell_y_min + offset_y

    return original_x, original_y

def clean_and_output_data():
    _time = time.time()
    cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
    # embs = pickle.load(open(Config.dataset_embs_file, 'rb'))
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00339/
    # download train.csv.zip and unzip it. rename train.csv to porto.csv
    dfraw = pd.read_csv(Config.root_dir + '/data/porto.csv')
    dfraw = dfraw.rename(columns = {"POLYLINE": "wgs_seq"})
    dfraw = dfraw[dfraw.MISSING_DATA == False]
    # print(dfraw.groupby("TAXI_ID").size())
    # print(dfraw["TIMESTAMP"].loc[0])
    # length requirement

    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
    logging.info('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj) ) # True: valid
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))
    total_points = dfraw['trajlen'].sum()
    print(f"num of total points: {total_points}.")
    # convert to Mercator
    # dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

    # logging.info('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    # dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq', 'time_each_point', 'traj_p']].reset_index(drop = True)
    # print(dfraw.columns)
    # dfraw.to_pickle(Config.dataset_file)
    # logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    # print(dfraw['traj_p'].loc[:10])
    return


def data_stats():
    cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
    # embs = pickle.load(open(Config.dataset_embs_file, 'rb'))
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00339/
    # download train.csv.zip and unzip it. rename train.csv to porto.csv
    dfraw = pd.read_csv(Config.root_dir + '/data/porto.csv')
    dfraw = dfraw.rename(columns = {"POLYLINE": "wgs_seq"})
    dfraw = dfraw[dfraw.MISSING_DATA == False]
    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)

    filtered_groups = dfraw.groupby('TAXI_ID').filter(lambda x: len(x) > 1000)
    grouped_trips = filtered_groups.groupby('TAXI_ID')
    trips_by_id = {taxi_id: group for taxi_id, group in grouped_trips}

    # Retrieve the first 5 unique IDs with more than 1000 trajectories
    unique_ids = filtered_groups['TAXI_ID'].unique()[:5]

    for taxi_id in unique_ids:
        # Extract only the rows for the current taxi ID
        data_subset = filtered_groups[filtered_groups['TAXI_ID'] == taxi_id]

        # Create a Plotly map
        fig = go.Figure()

        # Loop over each row in the data subset for the current taxi ID
        for index, row in data_subset.iterrows():
            polyline = row['wgs_seq']
            # Assuming polyline is a list of tuples (lat, lon)
            if polyline and all(len(point) == 2 for point in polyline if isinstance(point, list)):
                latitudes, longitudes = zip(*polyline)
                fig.add_trace(go.Scattermapbox(
                    lat=latitudes,
                    lon=longitudes,
                    mode='lines+markers',
                    marker=dict(size=5),
                    name=f'Trip index {index}'
                ))

        # Set the layout for the map
        fig.update_layout(
            mapbox={
                'style': "open-street-map",
                'center': {'lat': sum(latitudes) / len(latitudes), 'lon': sum(longitudes) / len(longitudes)},
                # Center the map around the average of the coordinates
                'zoom': 10  # Adjust zoom as needed
            },
            title=f"Trajectories for Taxi ID {taxi_id}"
        )

        # Save the figure to an HTML file
        file_name = f"Trajectories_{taxi_id}.html"
        fig.write_html(file_name)
        print(f"Saved {file_name}")
    # temp = dfraw.groupby("TAXI_ID").size().reset_index(name='Count')
    # data = {"ID": temp["TAXI_ID"].values, "num": temp["Count"].values}
    # print(data)

    # df = pd.DataFrame(data)
    # print((df['num'] > 1000).sum())

    # # Plotting the bar chart
    # plt.figure(figsize=(10, 6))
    # plt.bar(df['ID'], df['num'], color='blue')
    # plt.xlabel('Unique IDs')
    # plt.ylabel('Numbers')
    # plt.title('Number Distribution')
    # plt.show()


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
    trajs = pd.read_pickle(Config.dataset_file) # using test part only
    l = trajs.shape[0]
    n_query = 1000
    n_db = 100000
    test_idx = (int(l*0.8), int(l*0.8)+n_db)
    test_trajs = trajs[test_idx[0]: test_idx[1]]
    logging.info("Test trajs loaded.")

    # for varying db size
    def _raw_dataset():
        query_lst = [] # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.iteritems():
            if i < n_query:
                query_lst.append(np.array(v)[::2].tolist())
            db_lst.append(np.array(v)[1::2].tolist())
            i += 1

        output_file_name = Config.dataset_file + '_newsimi_raw.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump( (query_lst, db_lst) , fh, protocol = pickle.HIGHEST_PROTOCOL)
            logging.info("_raw_dataset done.")
        return

    # for varying downsampling rate
    def _downsample_dataset(rate):
        unrate = 1-rate # preserved rate 
        query_lst = [] # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.iteritems():
            if i < n_query:
                _q = np.array(v)[::2]
                _q_len = _q.shape[0]
                _idx = np.sort(np.random.choice(_q_len, math.ceil(_q_len*unrate), replace = False))
                query_lst.append(  _q[_idx].tolist()  )
            _db = np.array(v)[1::2]
            _db_len = _db.shape[0]
            _idx = np.sort(np.random.choice(_db_len, math.ceil(_db_len*unrate), replace = False))
            db_lst.append(  _db[_idx].tolist()  )
            i += 1

        output_file_name = Config.dataset_file + '_newsimi_downsampling_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump( (query_lst, db_lst) , fh, protocol = pickle.HIGHEST_PROTOCOL)
            logging.info("_downsample_dataset done. rate={}".format(rate))
        return

    # for varying distort rate
    def _distort_dataset(rate):
        query_lst = [] # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.iteritems():
            if i < n_query:
                _q = np.array(v)[::2]
                for _row in range(_q.shape[0]):
                    if random.random() < rate:
                        _q[_row] = _q[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
                query_lst.append(  _q.tolist()  )
            
            _db = np.array(v)[1::2]
            for _row in range(_db.shape[0]):
                if random.random() < rate:
                    _db[_row] = _db[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
            db_lst.append(  _db.tolist()  )
            i += 1

        output_file_name = Config.dataset_file + '_newsimi_distort_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump( (query_lst, db_lst) , fh, protocol = pickle.HIGHEST_PROTOCOL)
            logging.info("_distort_dataset done. rate={}".format(rate))
        return
    

    _raw_dataset()

    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _downsample_dataset(rate)

    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _distort_dataset(rate)

    return


# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name = 'hausdorff'):
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
    trains_simi = _simi_matrix(fn, trains) # [ [simi, simi, ... ], ... ]

    max_distance = max( max(map(max, trains_simi)), max(map(max, evals_simi)), max(map(max, tests_simi)) )

    _output_file = '{}_traj_simi_dict_{}.pkl'.format(Config.dataset_file, fn_name)
    with open(_output_file, 'wb') as fh:
        tup = trains_simi, evals_simi, tests_simi, max_distance
        pickle.dump(tup, fh, protocol = pickle.HIGHEST_PROTOCOL)
    
    logging.info("traj_simi_computation ends. @={:.3f}".format(time.time() - _time))
    return tup


def _normalization(lst_df):
    # lst_df: [df, df, df]
    xs = []
    ys = []
    for df in lst_df:
        for _, v in df.merc_seq.iteritems():
            arr = np.array(v)
            xs.append(arr[:,0])
            ys.append(arr[:,1])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    mean = np.array([xs.mean(), ys.mean()])
    std = np.array([xs.std(), ys.std()])

    for i in range(len(lst_df)):
        lst_df[i].merc_seq = lst_df[i].merc_seq.apply(lambda lst: ( (np.array(lst)-mean)/std ).tolist())

    return lst_df


def _get_simi_fn(fn_name):
    fn =  {'lcss': tdist.lcss, 'edr': tdist.edr, 'frechet': tdist.frechet,
            'discret_frechet': tdist.discret_frechet,
            'hausdorff': tdist.hausdorff, 'edwp': edwp}.get(fn_name, None)
    if fn_name == 'lcss' or fn_name == 'edr':
        fn = partial(fn, eps = Config.test_exp1_lcss_edr_epsilon)
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
            tasks.append( (fn, df, list(range(batch_size * i, batch_size * (i+1)))) )
        else:
            tasks.append( (fn, df, list(range(batch_size * i, l))) )
    

    num_cores = int(mp.cpu_count()) - 6
    assert num_cores > 0
    logging.info("pool.size={}".format(num_cores))
    pool = mp.Pool(num_cores)
    lst_simi = pool.starmap(_simi_comp_operator, tasks)
    pool.close()

    # extend lst_simi to matrix simi and pad 0s
    lst_simi = sum(lst_simi, [])
    for i, row_simi in enumerate(lst_simi):
        lst_simi[i] = [0]*(i+1) + row_simi
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
            simi_row.append( float(fn(t_i, t_j)) )
        simi.append(simi_row)
    logging.debug('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
                    .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi


def read_ft():
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle("../data/porto_20200_new")

    l = trajs.shape[0]
    print(l)
    # train_idx = (int(l*0), 200000)        # for porto
    # train_idx = (int(l * 0), 70000)  # for t-drive
    # train_idx = (int(l * 0), int(l*0.7))
    # train_idx = (int(l * 0), 35000)         # for geolife
    offset_idx = int(trajs.shape[0] * 0.7)  # use eval dataset
    print(offset_idx)
    df_trajs = trajs.iloc[offset_idx: offset_idx + 10000]
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
    df_trajs.to_csv("porto_ft.csv", index=False)


# nohup python ./preprocessing_porto.py &> ../result &
# if __name__ == '__main__':
#     logging.basicConfig(level = logging.DEBUG,
#                         format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
#                         handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'),
#                                     logging.StreamHandler()]
#                         )
#     Config.dataset = 'porto'
#     Config.post_value_updates()

    # data_stats()
    # clean_and_output_data()
    # init_cellspace()
    # generate_newsimi_test_dataset()
    # traj_simi_computation('lcss') # edr edwp discret_frechet hausdorff
    # read_ft()
print(Config.dataset)
dfraw = pd.read_csv(Config.root_dir + '/data/porto.csv')
dfraw = dfraw.rename(columns={"POLYLINE": "wgs_seq"})
dfraw = dfraw[dfraw.MISSING_DATA == False]
# print(dfraw.groupby("TAXI_ID").size())
# print(dfraw["TIMESTAMP"].loc[0])
# length requirement

dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
logging.info('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

# range requirement
dfraw['inrange'] = dfraw.wgs_seq.map(
    lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))  # True: valid
dfraw = dfraw[dfraw.inrange == True]
logging.info('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))
total_points = dfraw['trajlen'].sum()
print(f"num of total points: {total_points}.")

