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
from utils.data_loader import read_traj_dataset
from utils.traj import *
from utils import tool_funcs


def get_neighors(traj, nb, embs):
    """
    also need to add neighbor embeddings.
    """
    return [embs[nb[p]] for p in traj]

def _collate(trajs, cellspace, embs, neighbors):

    traj_cell = [merc2cell(t, cellspace) for t in trajs]
    traj_emb_cell = [embs[list(t)] for t in traj_cell]
    traj_emb_cell = [t.clone().detach() for t in traj_emb_cell]
    # Keep tensors on CPU in worker process to avoid CUDA multiprocessing issues
    traj_emb_cell = pad_sequence(traj_emb_cell, batch_first=True)  # CPU tensor
    num_points = torch.tensor(list(map(len, traj_cell)), dtype=torch.long)  # CPU tensor

    traj_o = [torch.tensor(generate_mask_tokens(t)) for t in trajs]
    traj_o = pad_sequence(traj_o, batch_first=True)  # CPU tensor
    

    neighbor_seq = [torch.stack(get_neighors(t, neighbors, embs)) for t in traj_cell]
    neighbor_seq = pad_sequence(neighbor_seq, batch_first=True)  # CPU tensor

    return traj_emb_cell, traj_o, num_points, neighbor_seq


class TrajJEPATrainer:

    def __init__(self):
        super(TrajJEPATrainer, self).__init__()
        self.embs = pickle.load(open(Config.dataset_embs_file, 'rb')).to('cpu').detach()  # tensor
        # self.embs = np.array(pickle.load(open("./data/embeddings.pkl", 'rb')))
        # print(self.embs[0].shape, len(self.embs))
        self.cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
        self.neighbors = pickle.load(open(Config.neighbours_file, 'rb'))
        # self.edge_xy = self.cellspace.all_neighbour_cell_pairs_permutated_optmized()
        # self.graph, self.edge_index, self.node_map = prepare_graph(Config, self.cellspace, self.edge_xy)
        # print(self.edge_xy)
        train_dataset, _, _ = read_traj_dataset(Config.dataset_file)
        self.num_train = len(train_dataset)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=Config.trajcl_batch_size,
                                           shuffle=False,
                                           num_workers=16,
                                           pin_memory=True, 
                                           persistent_workers=True,  
                                           drop_last=True,
                                           collate_fn=partial(_collate, cellspace=self.cellspace,
                                                              embs=self.embs, neighbors=self.neighbors))

        self.model = JEPA_base().to(Config.device)
        self.checkpoint_file = '{}/{}_{}{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix, Config.checkpoint_file,
                                                               Config.dumpfile_uniqueid)
        logging.info(f"Model will be saved to {self.checkpoint_file}")
        # self.model.load_checkpoint()
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
        if Config.dataset == "tokyo" or Config.dataset == "nyc":
            tmp_ckpt = torch.load("./exp/snapshots/porto_20200_new_t-jepa_porto_pretrain_motion_ca_noise_ep20_1e-4_mask234.pt")
            self.model.load_state_dict(tmp_ckpt['model_state_dict'])
            self.model.to(Config.device)

        for i_ep in range(Config.trajcl_training_epochs):

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
                # Move tensors to GPU (they were kept on CPU in worker process)
                cell_emb = cell_emb.to(Config.device)
                traj_o = traj_o.to(Config.device)
                num_points = num_points.to(Config.device)
                adj = adj.to(Config.device)

                context_out, target_emb = self.model(cell_emb, traj_o, num_points, adj, "train")
                loss = self.model.loss_fn(context_out, target_emb)

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
            if loss_ep_avg < best_loss_train:
                best_epoch = i_ep
                best_loss_train = loss_ep_avg
                bad_counter = 0
                self.save_checkpoint()
            else:
                bad_counter += 1

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
        if Config.pretrain_skip_eval:
            logging.info('[Eval] skipped for train-only pretrain dataset.')
            return

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
            if Config.dataset == "porto":
                start, n_skip, end = 20000, 20000, 100000
            elif Config.dataset == "beijing":
                start, n_skip, end = 2000, 2000, 10000
            elif Config.dataset == "geolife":
                start, n_skip, end = 2000, 2000, 10000
            elif Config.dataset == "tokyo":
                start, n_skip, end = 100, 100, 500
            elif Config.dataset == "nyc":
                start, n_skip, end = 30, 29, 147
            for n_db in range(start, end+1, n_skip):
                rank = torch.sum(torch.le(dists[:, 0:n_db].T, targets)).item() / len(q_lst)
                results.append(rank)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=T-JEPA,varying=dbsize,r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
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
                '[EXPFlag]task=newsimi,encoder=T-JEPA,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(vt, *results))
        return

    @torch.no_grad()
    def test(self):
        # 1. read best model
        # 2. read trajs from file, then -> embeddings
        # 3. run testing
        # n. varying db size, downsampling rates, and distort rates

        logging.info('[Test]start.')

        import time
        s = time.time()
        # if Config.dataset == "tokyo" or Config.dataset == "nyc" or Config.dataset == "ais_au" or Config.dataset == "migration":
        #     # print(1)
        #     tmp_ckpt = torch.load("./exp/snapshots/porto_20200_new_t-jepa_porto_pretrain_motion_ca_noise_ep20_1e-4_mask234.pt")
        #     self.model.load_state_dict(tmp_ckpt['model_state_dict'])
        #     self.model.to(Config.device)
        # else:
        #     self.load_checkpoint()
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
            if Config.dataset == "porto":
                start, n_skip, end = 20000, 20000, 100000
            elif Config.dataset == "beijing":
                start, n_skip, end = 2000, 2000, 10000
            elif Config.dataset == "geolife":
                start, n_skip, end = 2000, 2000, 10000
            elif Config.dataset == "tokyo":
                start, n_skip, end = 100, 100, 500
            elif Config.dataset == "nyc":
                start, n_skip, end = 30, 29, 147
            for n_db in range(start, end+1, n_skip):      # for porto
                rank = torch.sum(torch.le(dists[:, 0:n_db].T, targets)).item() / len(q_lst)
                results.append(rank)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=T-JEPA,varying=dbsize,r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
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
                '[EXPFlag]task=newsimi,encoder=T-JEPA,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(vt, *results))

        e = time.time()

        print(e-s)
        return

    @torch.no_grad()
    def test_merc_seq_to_embs(self, q_lst, db_lst):
        querys = []
        databases = []
        num_query = len(q_lst)  # 1000
        num_database = len(db_lst)  # 100000
        batch_size = num_query

        for i in range(num_database // batch_size):
            if i == 0:

                traj_emb1_cell, traj_offsets1, num_points1, adj1 \
                    = _collate(q_lst, self.cellspace, self.embs, self.neighbors)
                
                # Move tensors to GPU (they were kept on CPU in _collate)
                traj_emb1_cell = traj_emb1_cell.to(Config.device)
                traj_offsets1 = traj_offsets1.to(Config.device)
                num_points1 = num_points1.to(Config.device)
                adj1 = adj1.to(Config.device)

                trajs1_emb = self.model.interpret(traj_emb1_cell, traj_offsets1, num_points1, adj1)
                trajs1_emb = trajs1_emb.permute(1, 0, 2)
                trajs1_emb = torch.sum(trajs1_emb, 0)
                trajs1_emb = trajs1_emb / num_points1.unsqueeze(-1).expand(trajs1_emb.shape)
                querys.append(trajs1_emb)

            traj_emb2_cell, traj_offsets2, num_points2, adj2 \
                = _collate(db_lst[i * batch_size: (i + 1) * batch_size], self.cellspace, self.embs, self.neighbors)
            
            # Move tensors to GPU (they were kept on CPU in _collate)
            traj_emb2_cell = traj_emb2_cell.to(Config.device)
            traj_offsets2 = traj_offsets2.to(Config.device)
            num_points2 = num_points2.to(Config.device)
            adj2 = adj2.to(Config.device)

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



