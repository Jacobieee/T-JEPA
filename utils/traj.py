import sys
sys.path.append('..')
import numpy as np
import random
import math

from config import Config
from utils import tool_funcs
from utils.rdp import rdp
from utils.cellspace import CellSpace
from utils.tool_funcs import truncated_rand


def straight(src):
    return src


def simplify(src):
    # src: [[lon, lat], [lon, lat], ...]
    return rdp(src, epsilon = Config.traj_simp_dist)


def shift(src):
    return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in src]


def mask(src):
    l = len(src)
    arr = np.array(src)
    mask_idx = np.random.choice(l, int(l * Config.traj_mask_ratio), replace = False)
    return np.delete(arr, mask_idx, 0).tolist()


def subset(src):
    l = len(src)
    max_start_idx = l - int(l * Config.traj_subset_ratio)
    start_idx = random.randint(0, max_start_idx)
    end_idx = start_idx + int(l * Config.traj_subset_ratio)
    return src[start_idx: end_idx]


def get_aug_fn(name: str):
    return {'straight': straight, 'simplify': simplify, 'shift': shift,
            'mask': mask, 'subset': subset}.get(name, None)


# pair-wise conversion -- structural features and spatial feasures
def merc2cell2(src, cs: CellSpace):
    # convert and remove consecutive duplicates
    tgt = [ (cs.get_cellid_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]
    tgt, tgt_p = zip(*tgt)
    return tgt, tgt_p

def merc2cell(src, cs: CellSpace):
    # convert and remove consecutive duplicates
    tgt = [ cs.get_cellid_by_point(*p) for p in src]
    # tgt = [(p[:2], p[2:]) for p in src]
    # tgt = [(cs.get_cellid_by_point(*p[:2]), p[:2], p[-1]) for p in src]
    # print(tgt)
    # tgt = [v for i, v in enumerate(src[-1]) if i == 0 or v[0] != tgt[i-1][0]]
    # tgt, tgt_p, tgt_o = zip(*tgt)
    return tgt


def generate_spatial_features(src, cs: CellSpace):
    # src = [length, 2]
    tgt = []
    lens = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))

    for i in range(1, len(src) - 1):
        dist = (lens[i-1] + lens[i]) / 2
        dist = dist / (Config.trajcl_local_mask_sidelen / 1.414) # float_ceil(sqrt(2))

        radian = math.pi - math.atan2(src[i-1][0] - src[i][0],  src[i-1][1] - src[i][1]) \
                        + math.atan2(src[i+1][0] - src[i][0],  src[i+1][1] - src[i][1])
        radian = 1 - abs(radian) / math.pi

        x = (src[i][0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (src[i][1] - cs.y_min)/ (cs.y_max - cs.y_min)
        tgt.append( [x, y, dist, radian] )

    x = (src[0][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[0][1] - cs.y_min)/ (cs.y_max - cs.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0] )
    
    x = (src[-1][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[-1][1] - cs.y_min)/ (cs.y_max - cs.y_min)
    tgt.append( [x, y, 0.0, 0.0] )
    # tgt = [length, 4]
    return tgt


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length


def generate_mask_tokens(src):
    """
    Calculate turning angle and sinuosity for each point in the trajectory.
    
    Turning angle: Calculated same as generate_spatial_features (for points i=1 to len-2)
    Sinuosity: Actual Path Length / Euclidean Distance (from start to current point)
    
    Args:
        src: List of points [[lon, lat], [lon, lat], ...]
    
    Returns:
        List of [turning_angle, sinuosity] for each point.
        First and last points have turning_angle=0.0
    """
    if len(src) < 2:
        return [[0.0, 1.0] for _ in src]
    
    result = []
    cumulative_path_length = 0.0
    
    # First point: no turning angle, sinuosity = 1.0 (single point)
    result.append([0.0, 1.0])
    
    # Calculate segment lengths
    segment_lengths = []
    for p1, p2 in tool_funcs.pairwise(src):
        seg_len = tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
        segment_lengths.append(seg_len)
    
    # Process middle points (i = 1 to len-2)
    for i in range(1, len(src) - 1):
        # Calculate turning angle (same as generate_spatial_features)
        radian = math.pi - math.atan2(src[i-1][0] - src[i][0], src[i-1][1] - src[i][1]) \
                        + math.atan2(src[i+1][0] - src[i][0], src[i+1][1] - src[i][1])
        turning_angle = 1 - abs(radian) / math.pi
        
        # Calculate sinuosity: Actual Path Length / Euclidean Distance
        # Actual path length from start to current point
        cumulative_path_length += segment_lengths[i-1]
        
        # Euclidean distance from start to current point
        euclidean_dist = tool_funcs.l2_distance(src[0][0], src[0][1], src[i][0], src[i][1])
        
        # Sinuosity
        if euclidean_dist > 0:
            sinuosity = cumulative_path_length / euclidean_dist
        else:
            sinuosity = 1.0  # If points coincide, sinuosity = 1
        
        result.append([turning_angle, sinuosity])
    
    # Last point: no turning angle, sinuosity for entire trajectory
    cumulative_path_length += segment_lengths[-1] if segment_lengths else 0.0
    euclidean_dist_total = tool_funcs.l2_distance(src[0][0], src[0][1], src[-1][0], src[-1][1])
    
    if euclidean_dist_total > 0:
        sinuosity_total = cumulative_path_length / euclidean_dist_total
    else:
        sinuosity_total = 1.0
    
    result.append([0.0, sinuosity_total])
    
    return result

