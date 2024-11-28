import sys

sys.path.append('..')

import time
import logging
import pickle
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from functools import partial
from model.jepa import JEPA_base

from config import Config
from model.moco import MoCo
from model.dual_attention import DualSTB
from utils.data_loader import read_traj_dataset
from utils.traj import *
from utils import tool_funcs
from utils.preprocessing_porto import get_offset
from model.graph_func import *


def _collate(trajs, cellspace, embs):
    # traj_cell, point = zip(*[merc2cell2(t, cellspace) for t in trajs])
    # traj_cell = zip(*[merc2cell(t, cellspace) for t in trajs])
    # print(trajs)
    traj_cell = [merc2cell(t, cellspace) for t in trajs]
    # print(traj_cell)
    traj_emb_cell = [embs[list(t)] for t in traj_cell]
    traj_emb_cell = [torch.tensor(t) for t in traj_emb_cell]
    traj_emb_cell = pad_sequence(traj_emb_cell, batch_first=True).to(Config.device)
    # print(traj_cell)
    num_points = torch.tensor(list(map(len, traj_cell)), dtype=torch.long, device=Config.device)
    # print(num_points)
    """hhhh"""
    # emb_p = [torch.tensor(t) for t in p]
    # emb_p = pad_sequence(emb_p, batch_first=True).to(Config.device)
    # print(emb_p)
    # cell_ids = [list(t) for t in traj_cell]
    # cell_ids = [torch.tensor(t) for t in cell_ids]
    # cell_ids = pad_sequence(cell_ids, batch_first=True).to(Config.device)

    max_num_points = num_points.max().item()
    # traj_offsets = [torch.tensor(np.stack(list(traj_o))) for traj_o in point]
    # print(traj_offsets)
    # traj_offsets = pad_sequence(traj_offsets, batch_first=True).to(Config.device)

    paddings = torch.arange(max_num_points, device=Config.device)[None, :] >= num_points[:, None]
    inv_paddings = ~paddings

    B = traj_emb_cell.shape[0]
    # get trajectory adjacency matrix.
    adj_m, adj = get_adj_matrix(traj_cell, cellspace, embs, B, max_num_points, inv_paddings)
    adj_m = adj_m.to(Config.device)
    adj = adj.to(Config.device)
    # print(adj.shape)
    # adj = embs[list(t)] for adj in adj_m
    # return traj_emb_cell, traj_offsets.float(), num_points
    return traj_emb_cell, None, num_points, adj
    # return traj_emb_cell, emb_p.float(), num_points, adj
def _collate_old(trajs, cellspace, embs, node_map):
    # traj_cell, point = zip(*[merc2cell2(t[:, :2], cellspace) for t in trajs])
    # traj_cell = []
    # traj_offsets = []
    # ts = []
    traj_cell, offsets = zip(*[merc2cell(t, cellspace) for t in trajs])
    # print(traj_cell)
    # for t in trajs:
    #     # print(t.shape)
    #     # Process the first two columns of 't' through 'merc2cell'
    #     cells, traj_o = merc2cell(t, cellspace)
    #
    #     # Append the results to the respective lists
    #     traj_cell.append(cells)
    #     traj_offsets.append(traj_o)
        # print(f"cell: {cells}")
        # print(f"traj_o: {traj_o}")
        # print(f"time: {time}")
        # print(t[:, -1])
        # Directly append the last column of 't' to 'last_elements'
        # ts.append(time)

    # traj_emb_cell = [embs[list(t)] for t in traj_cell]

    # [print(coord) for coord in traj_cell]
    # print([[np.stack(list(traj_o))[:, :2] for traj_o in offsets]])

    node_index = [np.stack(list(traj_o))[:, :2] for traj_o in offsets]
    # print(node_index)
    node_index = [arr.astype(int) for arr in node_index]
    node_index = [list(map(tuple, array)) for array in node_index]
    # print(node_index)

    traj_emb_cell = [embs[list(node_map[n] for n in node)] for node in node_index]
    # traj_emb_cell = [embs[np.stack(list(t)[:2])] for t in offsets]

    traj_emb_cell = [torch.tensor(t) for t in traj_emb_cell]
    num_points = torch.tensor(list(map(len, traj_cell)), dtype=torch.long, device=Config.device)
    # max_num_points = num_points.max().item()
    traj_emb_cell = pad_sequence(traj_emb_cell, batch_first=True).to(Config.device)
    # print(traj_emb_cell.shape)
    traj_offsets = [torch.tensor(np.stack(list(traj_o))) for traj_o in offsets]
    # print(traj_offsets)
    traj_offsets = pad_sequence(traj_offsets, batch_first=True).to(Config.device)
    # ts = [torch.tensor(list(t)) for t in ts]
    # ts = pad_sequence(ts, batch_first=True).to(Config.device)
    # paddings = torch.arange(max_num_points, device=Config.device)[None, :] >= num_points[:, None]
    # inv_paddings = ~paddings

    # B = traj_emb_cell.shape[1]
    # get trajectory adjacency matrix.
    # adj_m = get_adj_matrix(traj_cell, cellspace, B, max_num_points, inv_paddings)
    return traj_emb_cell, traj_offsets.float(), num_points
    # return traj_offsets.float(), num_points
    # return traj_emb_cell, point, ts, num_points, adj_m


class TrajJEPATrainer:

    def __init__(self):
        super(TrajJEPATrainer, self).__init__()
        self.embs = pickle.load(open(Config.dataset_embs_file, 'rb')).to('cpu').detach()  # tensor
        # self.embs = np.array(pickle.load(open("./data/embeddings.pkl", 'rb')))
        # print(self.embs[0].shape, len(self.embs))
        self.cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
        # self.edge_xy = self.cellspace.all_neighbour_cell_pairs_permutated_optmized()
        # self.graph, self.edge_index, self.node_map = prepare_graph(Config, self.cellspace, self.edge_xy)
        # print(self.edge_xy)
        train_dataset, _, _ = read_traj_dataset(Config.dataset_file)
        self.num_train = len(train_dataset)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=Config.trajcl_batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           drop_last=True,
                                           collate_fn=partial(_collate, cellspace=self.cellspace,
                                                              embs=self.embs))

        self.model = JEPA_base().to(Config.device)
        # print(self.model.parameters())
        # self.model = nn.DataParallel(self.model, device_ids=[0,1])
        # self.model.to(Config.device)
        self.checkpoint_file = '{}/{}_Traj-JEPA_adj_fuse_new{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix,
                                                               Config.dumpfile_uniqueid)
        print(f"Model will be saved to {self.checkpoint_file}")
        self.ema = (0.996, 1.0)
        self.ipe_scale = 1.0
    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={:.0f}".format(training_starttime))
        torch.autograd.set_detect_anomaly(True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.trajcl_training_lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.trajcl_training_lr_degrade_step,
                                                    gamma=Config.trajcl_training_lr_degrade_gamma)
        training_steps = self.num_train / Config.trajcl_batch_size
        momentum_scheduler = (self.ema[0] + i * (self.ema[1] - self.ema[0]) / (training_steps * Config.trajcl_training_epochs * self.ipe_scale)
                              for i in range(int(training_steps * Config.trajcl_training_epochs * self.ipe_scale) + 1))

        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = Config.trajcl_training_bad_patience
        """"""
        # tmp_ckpt = torch.load("./exp/snapshots/porto_20200_new_Traj-JEPA_adj_fuse.pt")
        # self.model.load_state_dict(tmp_ckpt['model_state_dict'])
        # self.model.to(Config.device)
        # - 0.996
        # - 1.0

        for i_ep in range(Config.trajcl_training_epochs):
            # if i_ep == 1 or i_ep == 2:
            self.eval()

            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []

            self.model.train()

            _time_batch_start = time.time()
            for i_batch, batch in enumerate(self.train_dataloader):
                _time_batch = time.time()
                optimizer.zero_grad()

                cell_emb, traj_o, num_points, adj = batch

                # map, nodes = train_gsage(Config, self.cellspace, self.edge_xy)
                # node_index = [[nodes[(int(coord[0].item()), int(coord[1].item()))] for coord in traj] for traj in traj_o[:, :, :2]]
                # cell_emb = map[node_index]
                # cell_emb = torch.tensor(cell_emb, dtype=torch.long).to(Config.device)
                context_out, target_emb = self.model(cell_emb, traj_o, num_points, adj, "train")
                # print(target_emb)
                loss = self.model.loss_fn(context_out, target_emb)
                # print(context_out[0, :, 0, :])
                # print(target_emb[0, :, 0, :])

                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_ep.append(loss.item())
                # EMA.
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for context, target in zip(self.model.context_encoder.parameters(),
                                               self.model.target_encoder.parameters()):
                        target.data.mul_(m).add_((1.-m) * context.detach().data)

                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())

                if i_batch % 100 == 0 and i_batch:
                    logging.debug("[Training] ep-batch={}-{}, loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                                  .format(i_ep, i_batch, loss.item(), time.time() - _time_batch_start,
                                          tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            scheduler.step()  # decay before optimizer when pytorch < 1.1

            loss_ep_avg = tool_funcs.mean(loss_ep)
            logging.info("[Training] ep={}: avg_loss={:.3f}, @={:.3f}/{:.3f}, gpu={}, ram={}" \
                         .format(i_ep, loss_ep_avg, time.time() - _time_ep, time.time() - training_starttime,
                                 tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # early stopping
            self.save_checkpoint()
            # if loss_ep_avg < best_loss_train:
            #     best_epoch = i_ep
            #     best_loss_train = loss_ep_avg
            #     bad_counter = 0
            #     self.save_checkpoint()
            # else:
            #     bad_counter += 1

            if bad_counter == bad_patience or (i_ep + 1) == Config.trajcl_training_epochs:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                             .format(time.time() - training_starttime, best_epoch, best_loss_train))
                break

        return {'enc_train_time': time.time() - training_starttime, \
                'enc_train_gpu': training_gpu_usage, \
                'enc_train_ram': training_ram_usage}



    @torch.no_grad()
    def eval(self):
        # 1. read best model
        # 2. read trajs from file, then -> embeddings
        # 3. run testing
        # n. varying db size, downsampling rates, and distort rates

        logging.info('[Eval]start.')
        # self.load_checkpoint()
        self.model.eval()

        # varying db size
        with open(Config.dataset_file + '_newsimi_raw.pkl', 'rb') as fh:
            q_lst, db_lst = pickle.load(fh)
            querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
            # print(querys.shape)
            # print(databases.shape)
            dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
            targets = torch.diag(dists)  # [1000]
            results = []
            for n_db in range(20000, 100001, 20000):
                rank = torch.sum(torch.le(dists[:, 0:n_db].T, targets)).item() / len(q_lst)
                results.append(rank)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=TrajCL,varying=dbsize,r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(*results))

        # varying downsampling; varying distort
        for vt in ['downsampling', 'distort']:
            results = []
            for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
                with open(Config.dataset_file + '_newsimi_' + vt + '_' + str(rate) + '.pkl', 'rb') as fh:
                    q_lst, db_lst = pickle.load(fh)
                    querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
                    dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
                    targets = torch.diag(dists)  # [1000]
                    result = torch.sum(torch.le(dists.T, targets)).item() / len(q_lst)
                    results.append(result)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=TrajCL,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(vt, *results))
        return

    @torch.no_grad()
    def test(self):
        # 1. read best model
        # 2. read trajs from file, then -> embeddings
        # 3. run testing
        # n. varying db size, downsampling rates, and distort rates

        logging.info('[Test]start.')
        self.load_checkpoint()
        self.model.eval()

        # varying db size
        with open(Config.dataset_file + '_newsimi_raw.pkl', 'rb') as fh:
            q_lst, db_lst = pickle.load(fh)
            querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
            # print(querys.shape)
            # print(len(databases))
            dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
            targets = torch.diag(dists)  # [1000]
            results = []
            # for n_db in range(20000, 100001, 20000):      # for porto
            for n_db in range(2000, 10001, 2000):           # for tdrive
                rank = torch.sum(torch.le(dists[:, 0:n_db].T, targets)).item() / len(q_lst)
                results.append(rank)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=TrajCL,varying=dbsize,r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(*results))

        # varying downsampling; varying distort
        for vt in ['downsampling', 'distort']:
            results = []
            for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
                with open(Config.dataset_file + '_newsimi_' + vt + '_' + str(rate) + '.pkl', 'rb') as fh:
                    q_lst, db_lst = pickle.load(fh)
                    querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
                    dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
                    targets = torch.diag(dists)  # [1000]
                    result = torch.sum(torch.le(dists.T, targets)).item() / len(q_lst)
                    results.append(result)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=TrajCL,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(vt, *results))
        return

    @torch.no_grad()
    def test_merc_seq_to_embs(self, q_lst, db_lst):
        querys = []
        databases = []
        num_query = len(q_lst)  # 1000
        num_database = len(db_lst)  # 100000
        batch_size = num_query

        # def convert_trajectory(trajectory, cellspace):
        #     """Convert a trajectory to a list of (cell_id, offset_x, offset_y)."""
        #     return [get_offset(x, y, cellspace) for x, y in trajectory]

        # new_q_lst = []
        # for traj in q_lst:
        #     new_trajectory = []
        #     converted_trajectory = convert_trajectory(traj, self.cellspace)
        #     # Combine each original and converted point
        #     for orig, offset in zip(traj, converted_trajectory):
        #         # Combine orig (x, y) with conv (x*cellspace, y*cellspace)
        #         combined_point = orig + list(offset)
        #         new_trajectory.append(combined_point)
        #     new_q_lst.append(new_trajectory)
        # print(new_q_lst[0])

        # new_db_lst = []
        # for traj in db_lst:
        #     new_trajectory = []
        #     converted_trajectory = convert_trajectory(traj, self.cellspace)
        #     # Combine each original and converted point
        #     for orig, offset in zip(traj, converted_trajectory):
        #         # Combine orig (x, y) with conv (x*cellspace, y*cellspace)
        #         combined_point = orig + list(offset)
        #         new_trajectory.append(combined_point)
        #     new_db_lst.append(new_trajectory)
        # print(q_lst)
        for i in range(num_database // batch_size):
            if i == 0:

                traj_emb1_cell, traj_offsets1, num_points1, adj1 \
                    = _collate(q_lst, self.cellspace, self.embs)

                # map, nodes = train_gsage(Config, self.cellspace, self.edge_xy)
                # node_index = [[nodes[(int(coord[0].item()), int(coord[1].item()))] for coord in traj] for traj in
                #               traj_offsets1[:, :, :2]]
                # cell_emb1 = map[node_index]
                trajs1_emb = self.model.interpret(traj_emb1_cell, traj_offsets1, num_points1, adj1)
                trajs1_emb = trajs1_emb.permute(1, 0, 2)
                trajs1_emb = torch.sum(trajs1_emb, 0)
                trajs1_emb = trajs1_emb / num_points1.unsqueeze(-1).expand(trajs1_emb.shape)
                querys.append(trajs1_emb)

            traj_emb2_cell, traj_offsets2, num_points2, adj2 \
                = _collate(db_lst[i * batch_size: (i + 1) * batch_size], self.cellspace, self.embs)

            # map, nodes = train_gsage(Config, self.cellspace, self.edge_xy)
            # node_index = [[nodes[(int(coord[0].item()), int(coord[1].item()))] for coord in traj] for traj in
            #               traj_offsets2[:, :, :2]]
            # cell_emb2 = map[node_index]
            trajs2_emb = self.model.interpret(traj_emb2_cell, traj_offsets2, num_points2, adj2)
            trajs2_emb = trajs2_emb.permute(1, 0, 2)
            trajs2_emb = torch.sum(trajs2_emb, 0)
            trajs2_emb = trajs2_emb / num_points2.unsqueeze(-1).expand(trajs2_emb.shape)
            databases.append(trajs2_emb)

        querys = torch.cat(querys)  # tensor; traj embeddings
        databases = torch.cat(databases)
        return querys, databases

    def dump_embeddings(self):
        return

    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict()},
                   self.checkpoint_file)
        return

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(Config.device)

        return



