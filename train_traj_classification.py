import math
import logging
import random
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from config import Config
import pandas as pd
# from traj import *
from model.graph_func import *
import sys
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from utils.cellspace import CellSpace


import argparse
from model.jepa import JEPA_base
from datetime import datetime, timezone, timedelta

"""
tool function can't be loaded
"""

def log_file_name():
    dt = datetime.now(timezone(timedelta(hours=8)))
    return dt.strftime("%Y%m%d_%H%M%S") + '.log'


"""
baseline_jepa -- dataset
"""
class TrajDataset(Dataset):
    def __init__(self, data):
        # data: DataFrame
        self.data = data.reset_index(drop=True)
        self.label_encoder = LabelEncoder()
        self.data['Mode'] = self.label_encoder.fit_transform(self.data['Mode'])
            
        if self.data.isnull().values.any():
            raise ValueError("-----------Dataset contains NaN values.")
            
    def __getitem__(self, index):
        traj = np.array(self.data.loc[index].merc_seq)
        mode = self.data.loc[index].Mode
        return traj, mode

    def __len__(self):
        return int(self.data.shape[0])
    
    def get_label_encoder(self):
        return self.label_encoder

def read_traj_dataset(file_path):
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle(file_path)

    l = trajs.shape[0]
    # train_idx = (int(l*0), 200000)
    # train_idx = (int(l * 0), 70000)
    # train_idx = (int(l * 0), int(l*0.7))
    train_idx = (int(l * 0), 35000)
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))

    _train = TrajDataset(trajs[train_idx[0]: train_idx[1]])
    _eval = TrajDataset(trajs[eval_idx[0]: eval_idx[1]])
    _test = TrajDataset(trajs[test_idx[0]: test_idx[1]])

    logging.info('[Load traj dataset] END. @={:.0f}, #={}({}/{}/{})' \
                .format(time.time() - _time, l, len(_train), len(_eval), len(_test)))
    return _train, _eval, _test


"""
dataloader part
"""
def merc2cell(src, cs: CellSpace):
    # convert and remove consecutive duplicates
    tgt = [ cs.get_cellid_by_point(*p) for p in src]
    # tgt = [(p[:2], p[2:]) for p in src]
    # tgt = [(cs.get_cellid_by_point(*p[:2]), p[:2], p[-1]) for p in src]
    # print(tgt)
    # tgt = [v for i, v in enumerate(src[-1]) if i == 0 or v[0] != tgt[i-1][0]]
    # tgt, tgt_p, tgt_o = zip(*tgt)
    return tgt

def _collate(batch, cellspace, embs):
    trajs, modes = zip(*batch)
    traj_cell = [merc2cell(t, cellspace) for t in trajs]
    traj_emb_cell = [embs[list(t)] for t in traj_cell]
    # traj_emb_cell = [torch.tensor(t) for t in traj_emb_cell]
    traj_emb_cell = [t.clone().detach() for t in traj_emb_cell]
    traj_emb_cell = pad_sequence(traj_emb_cell, batch_first=True).to(Config.device)

    num_points = torch.tensor(list(map(len, traj_cell)), dtype=torch.long, device=Config.device)
    max_num_points = num_points.max().item()
    
    paddings = torch.arange(max_num_points, device=Config.device)[None, :] >= num_points[:, None]
    inv_paddings = ~paddings

    B = traj_emb_cell.shape[0]
    adj_m, adj = get_adj_matrix(traj_cell, cellspace, embs, B, max_num_points, inv_paddings)
    adj_m = adj_m.to(Config.device)
    adj = adj.to(Config.device)

    modes = torch.tensor(modes, dtype=torch.long).to(Config.device)
    
    return traj_emb_cell, None, num_points, adj, modes

"""
TrajSimiClass Frame
"""
class TrajSimiClassification(nn.Module):
    def __init__(self, dim_in):
        super(TrajSimiClassification, self).__init__()
        hidden_dim = 64
        # self.lstm = nn.LSTM(dim_in, hidden_dim, batch_first=True)
        self.enc = nn.Sequential(nn.Linear(256, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 4))
    
    def forward(self, trajs):
        # print('=====check trajs=====', trajs.shape)
        # _, (h_n, _) = self.lstm(trajs)
        # h_n = h_n.squeeze(0)
        h_n = trajs[:,-1,:]
        out = self.enc(h_n)
        
        return F.softmax(out, dim=1), out


class TrajSimiClass:
    def __init__(self, encoder):
        super(TrajSimiClass, self).__init__()
        self.encoder = encoder

        self.checkpoint_filepath = '{}/{}_trajsimi_{}_{}_best_alltune{}.pt' \
                                    .format(Config.checkpoint_dir, Config.dataset_prefix, \
                                            Config.trajsimi_encoder_name, \
                                            Config.trajsimi_measure_fn_name, \
                                            Config.dumpfile_uniqueid)
        
        self.cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
        self.cellembs = pickle.load(open(Config.dataset_embs_file, 'rb')).to(Config.device) # tensor

        """
        train/val/test datasets returned from read_traj_dataset function are all instance of the TrajDataset class

        """
        train_dataset, eval_dataset, _ = read_traj_dataset(Config.dataset_file)
        self.num_train = len(train_dataset)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=Config.trajcl_batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           drop_last=True,
                                           collate_fn=lambda x: _collate(x, self.cellspace, self.cellembs))
        
        self.eval_dataloader = DataLoader(eval_dataset,
                                          batch_size=Config.trajcl_batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          drop_last=True,
                                          collate_fn=lambda x: _collate(x, self.cellspace, self.cellembs))

        _seq_embedding_dim = Config.seq_embedding_dim
        self.trajsimiclassification = TrajSimiClassification(_seq_embedding_dim)
        self.trajsimiclassification.to(Config.device)

        """
        CE loss for classification
        """
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(Config.device)
        
        self.optimizer = torch.optim.Adam( [ \
                        {'params': self.trajsimiclassification.parameters(), \
                            'lr': Config.trajsimi_learning_rate, \
                            'weight_decay': Config.trajsimi_learning_weight_decay} ] )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 10, gamma = 0.5)

    def train(self):
        for i_ep in range(Config.trajsimi_epoch):
            _time_ep = time.time()
            
            self.trajsimiclassification.train()
            self.encoder.eval()

            train_losses = []
            all_preds = []
            all_labels = []


            for i_batch, batch in enumerate(self.train_dataloader):

                cell_emb, traj_o, num_points, adj, mode = batch
                # print('======check input========')
                embs = self.encoder.interpret(cell_emb, traj_o, num_points, adj)
                # print('==========check output=======', embs.shape, mode.shape, mode[0:3])
                outs, logits = self.trajsimiclassification(embs)
                # print('==========check output=======', outs.shape)
                loss_train = self.criterion(logits, mode)

                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

                train_losses.append(loss_train.item())
                _, pres = torch.max(outs, 1)
                all_preds.extend(pres.detach().cpu().numpy())
                all_labels.extend(mode.detach().cpu().numpy())

            self.scheduler.step() # decay before optimizer when pytorch < 1.1

            # i_ep
            print('===========================================================')
            print('training. i_ep={}, loss={:.4f}, cost {:.3f} secs'.format(i_ep, np.mean(train_losses), time.time()-_time_ep))
            # logging.info("training. i_ep={}, loss={:.4f}, cost {:.3f} secs" \
            #             .format(i_ep, np.mean(train_losses), time.time()-_time_ep))
            # print('======== check ========', len(all_preds), len(all_labels))
            acc, prec, recall, f1 = self.matrix_eval(all_preds, all_labels)
            print("matrix. accuracy={}, prec={}, recall={}, f1={}".format(acc, prec, recall, f1))
            # logging.info("matrix. accuracy={}, prec={}, recall={}, f1={}" \
            #             .format(acc, prec, recall, f1))
            
            self.eval()
            
    def eval(self): 
        self.encoder.eval()
        self.trajsimiclassification.eval()
        eval_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i_batch, batch in enumerate(self.eval_dataloader):
                cell_emb, traj_o, num_points, adj, mode = batch
                embs = self.encoder.interpret(cell_emb, traj_o, num_points, adj)
                outs, logits = self.trajsimiclassification(embs)
                
                # 检查模型输出
                if torch.isnan(logits).any():
                    print(f'Batch {i_batch}: Found NaN in logits')
                if torch.isnan(outs).any():
                    print(f'Batch {i_batch}: Found NaN in outs')
                
                loss_eval = self.criterion(logits, mode)
                
                if torch.isnan(loss_eval).any():
                    print(f'Batch {i_batch}: Found NaN in loss_eval')
                
                eval_losses.append(loss_eval.item())
                _, pres = torch.max(outs, 1)
                all_preds.extend(pres.detach().cpu().numpy())
                all_labels.extend(mode.detach().cpu().numpy())

        # 评估日志
        acc, prec, recall, f1 = self.matrix_eval(all_preds, all_labels)
        print('-----------------------------------------------------')
        print('evaluation. loss={:.4f}'.format(np.mean(eval_losses)))
        print("matrix. accuracy={}, prec={}, recall={}, f1={}".format(acc, prec, recall, f1))

    def matrix_eval(self, preds, labels):
        # print('======= check inside matrix =====', preds[0:3], labels[0:3])
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        return accuracy, precision, recall, f1
    

def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description = "train_traj_classification.py")
    parser.add_argument('--dumpfile_uniqueid', type = str, help = 'see config.py')
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')
    
    parser.add_argument('--trajsimi_measure_fn_name', type = str, help = '')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def main():
    enc_name = Config.trajsimi_encoder_name
    fn_name = Config.trajsimi_measure_fn_name
    # metrics = tool_funcs.Metrics()

    jepa = JEPA_base()
    jepa.load_checkpoint()
    jepa.to(Config.device)

    print("freezing encoder")
    # Freeze all parameters first
    for name, param in jepa.named_parameters():
        param.requires_grad = False
        print(f"{name} is frozen.")

    task = TrajSimiClass(jepa)
    task.train()
    # metrics.add(task.train())

    logging.info('[EXPFlag]model={},dataset={},fn={}'.format( \
                enc_name, Config.dataset_prefix, fn_name))
    return


# nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name hausdorff &> result &
if __name__ == '__main__':
    Config.update(parse_args())

    logging.basicConfig(level = logging.DEBUG if Config.debug else logging.INFO,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/log/'+log_file_name(), mode = 'w'), 
                        logging.StreamHandler()]
            )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    main()
