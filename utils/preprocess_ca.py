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
from utils import tool_funcs
from utils.cellspace import CellSpace
from utils.tool_funcs import lonlat2meters
from model.node2vec_ import train_node2vec
from utils.edwp import edwp
from utils.data_loader import read_trajsimi_traj_dataset
from utils.traj import *

def inrange(lon, lat):
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True


def join_data(f1, f2, f3):
    files = [f1, f2, f3]
    df = [pd.read_csv(file) for file in files]
    df = pd.concat(df, ignore_index=True)
    df.to_csv('gowalla.csv', index=False)


def get_mbr(f):
    df = pd.read_csv(f)
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        print("Please check the column names for Latitude and Longitude.")
    else:
        # Calculate minimum and maximum for Latitude
        min_latitude = df["Latitude"].min()
        max_latitude = df["Latitude"].max()

        # Calculate minimum and maximum for Longitude
        min_longitude = df["Longitude"].min()
        max_longitude = df["Longitude"].max()

        print(f"Minimum Latitude: {min_latitude}")
        print(f"Maximum Latitude: {max_latitude}")
        print(f"Minimum Longitude: {min_longitude}")
        print(f"Maximum Longitude: {max_longitude}")

# def process_ca(file):


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'),
                                    logging.StreamHandler()]
                        )
    Config.dataset = 'gowalla'
    Config.post_value_updates()

    # join_data("../ca/train_sample.csv", "../ca/validate_sample_with_traj.csv", "../ca/test_sample_with_traj.csv")
    get_mbr("../data/gowalla.csv")