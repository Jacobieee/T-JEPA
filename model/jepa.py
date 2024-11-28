import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from transformers import BertModel, BertConfig
from config import Config
from model.dual_attention import *
from torch.nn.utils.rnn import pad_sequence
from x_transformers import Decoder

from model.base_bert import BaseBERT, init_weights
from model.transformer import TransformerEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 201):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, token_embedding.size(1), :])


class JEPA_base(nn.Module):
    def __init__(self, M=4):
        super(JEPA_base, self).__init__()
        self.M = M
        # self.target_encoder = TransformerEncoder(input_dim=4,
        #                                          embed_dim=Config.seq_embedding_dim,
        #                                          num_heads=Config.attn_head,
        #                                          ff_dim=Config.hidden_dim,
        #                                          num_layers=Config.attn_layer)
        #
        # self.context_encoder = TransformerEncoder(input_dim=4,
        #                                          embed_dim=Config.seq_embedding_dim,
        #                                          num_heads=Config.attn_head,
        #                                          ff_dim=Config.hidden_dim,
        #                                          num_layers=Config.attn_layer)
        self.target_encoder = BaseBERT(Config, input_dim=2,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)
        # self.target_encoder.load_state_dict(torch.load('/mnt/data728/lihuan/my_proj/encoder_weights.pth'))
        self.context_encoder = BaseBERT(Config, input_dim=2,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)

        # self.context_encoder.load_state_dict(torch.load('/mnt/data728/lihuan/my_proj/encoder_weights.pth'))

        self.norm = nn.LayerNorm(Config.seq_embedding_dim)
        self.pe = nn.Parameter(torch.randn(1, Config.max_traj_len, Config.cell_embedding_dim))
        self.latent_var = nn.Parameter(torch.randn(1, 1, Config.cell_embedding_dim))
        self.predictor = Decoder(dim=Config.cell_embedding_dim, depth=2, heads=8)

        self.proj_out = nn.Linear(Config.seq_embedding_dim, Config.seq_embedding_dim)
        self.loss = nn.SmoothL1Loss(reduction='none')
        #
        for m in self.context_encoder.modules():
            init_weights(m)
        for m in self.target_encoder.modules():
            init_weights(m)
        #
        for m in self.predictor.modules():
            init_weights(m)


    @staticmethod
    def mask_tensor(x, src_padding_mask, mask_ratio, succ_prob=0.5):
        """
        Mask a tensor with a given mask ratio, considering the padding mask and
        a probability to apply successive masking.

        Args:
        - x (torch.Tensor): input embedding with shape [L, B, d].
        - src_padding_mask (torch.Tensor): Padding mask with shape [batch, len], True for padded positions.
        - mask_ratio (float): The ratio of len to be masked.
        - succ_prob (float): The probability of applying successive masking.

        Returns:
        - torch.Tensor: target points, target point indices.
        """
        # traj_emb, traj_emb_p, src_padding_mask = x
        B, L, _ = x.shape
        # Initialize a mask for all positions as True (indicating inclusion)
        mask = torch.zeros((B, L), dtype=torch.bool, device=x.device)
        mask_indices = []  # To store mask indices for each batch
        for b in range(B):
            # calculate the non-padding part.
            non_padded_len = L - src_padding_mask[b].sum().item()
            num_mask = int(non_padded_len * mask_ratio)
            # print(f"masking {num_mask} points.")
            # Ensure num_mask is not zero to proceed with masking
            if num_mask > 0:
                if torch.rand(1).item() < succ_prob:
                    # print("successive padding.")
                    # Successive masking
                    start = torch.randint(0, non_padded_len - num_mask + 1, (1,)).item()
                    indices = torch.arange(start, start + num_mask, device=x.device)
                    mask[b, indices] = True
                else:
                    # print("random padding.")
                    # Random masking
                    non_padded_indices = torch.nonzero(~src_padding_mask[b], as_tuple=False).squeeze()
                    # if 0 in non_padded_indices:
                    #     non_padded_indices = non_padded_indices[non_padded_indices != 0]
                    indices = non_padded_indices[torch.randperm(len(non_padded_indices))[:num_mask]]
                    mask[b, indices] = True

                mask_indices.append(indices)
        # Invert the source padding mask and overlay the target mask.
        # so that the masked parts are False.
        # inv_src_padding_mask = ~src_padding_mask.transpose(0, 1)
        # combined_mask = mask & inv_src_padding_mask
        # print(combined_mask)
        # Apply the combined mask to traj_emb
        # masked_x = torch.where(combined_mask.unsqueeze(-1), torch.tensor(0.0, device=x.device, dtype=x.dtype), x)
        masked_x = x * mask.unsqueeze(-1).to(x.dtype)
        # mask = torch.stack(mask, dim=1)

        return masked_x, mask, mask_indices
        # return mask_indices     # [B, len]


    @torch.no_grad()
    def target_seg(self, x, Config):
        """
        All we need is the target input, and the M * B masked segments.
        """
        cell_emb, traj_o, num_points, src_padding_mask, adj = x
        self.target_encoder.eval()
        # generate embeddings by target encoder.
        x_emb = self.target_encoder(**{
            'cell_emb': cell_emb,
            'traj_o': traj_o,
            'src_padding_mask': src_padding_mask,
            'adj': adj
        })  # [B, L, d]
        # print(x_emb)
        # x_emb = self.norm(x_emb)
        mask_ratio = Config.mask_ratio
        # get a random mask ratio and determine if the mask will be successive.
        idx = torch.randint(low=0, high=len(mask_ratio), size=(1,))
        r = mask_ratio[idx]
        # print(r)
        # target_segments = []
        all_masks = []      # masks for [M, B, L]
        all_mask_x = []     # masked padded sequence for [M, B, L, d]
        for m in range(self.M):
            # the mask indices will be successive or not for a whole batch.
            mask_x, masks, mask_indices = self.mask_tensor(x_emb, src_padding_mask, r, Config.succ_prob)
            all_masks.append(masks)
            all_mask_x.append(mask_x)

        all_masks = torch.stack(all_masks, dim=0)
        all_mask_x = torch.stack(all_mask_x, dim=0)

        return all_mask_x, all_masks


    def context_seg(self, x, Config, all_masks):
        cell_emb, traj_o, src_padding_mask, adj = x
        lb = Config.context_lb  # Lower bound ratio for sampling

        # Determine the number of sequence elements to sample based on the lower bound
        B, L, d = cell_emb.shape
        # Generate a unique sampling ratio for each trajectory in the batch within [lb, 1]
        ratios = torch.rand((B,), device=cell_emb.device).clamp(min=lb, max=1.0)
        # determine the number of samples kept considering the padding masks.
        non_padded_lens = L - src_padding_mask.sum(dim=1)
        num_samples = (non_padded_lens.float() * ratios).long()

        sampled_indices_list = [torch.empty(0, dtype=torch.long, device=cell_emb.device) for _ in range(B)]
        aug_sampled_emb = torch.zeros((self.M, B, L, d), device=cell_emb.device)
        # aug_sampled_traj_o = torch.zeros((self.M, B, L, 2), device=cell_emb.device)
        aug_sampled_adj = torch.zeros((self.M, B, L, 9, d), device=cell_emb.device)
        aug_src_padding_mask = torch.ones((self.M, B, L), dtype=torch.bool, device=cell_emb.device)
        # formulate the context.
        for b in range(B):
            # is true if the point is not on a position of a padding mask.
            valid_indices = torch.arange(L)[~src_padding_mask[b]].to(cell_emb.device)
            # sample selected indices according to ratio.
            sampled_indices = valid_indices[torch.randperm(valid_indices.size(0))[:num_samples[b]]]
            # print(f"{num_samples[b]} sample indices {sampled_indices} in traj. of length {torch.sum(~src_padding_mask[b, :])} in pm {src_padding_mask.shape[-1]}")
            # print(torch.sum(~src_padding_mask[b, :]))
            # print(f"indices{sampled_indices}")
            # assert torch.max(sampled_indices) < torch.sum(~src_padding_mask[b, :])
            sampled_indices_list[b] = sampled_indices
            # find overlaps
            curr_src_pad = src_padding_mask[b, :]
            for m in range(self.M):
                masks = all_masks[m, b, :]
                # print(f"trg mask: {torch.where(masks==1)}")
                # print(~sampled_indices)
                sample_mask = torch.zeros(L, dtype=torch.bool, device=cell_emb.device)
                sample_mask[sampled_indices] = 1
                # overlap_mask = ~sampled_indices.unsqueeze(1).eq(masks).any(dim=-1)
                overlap_mask = sample_mask & masks
                sample_mask[overlap_mask.bool()] = 0
                # print(f"overlaps: {torch.where(overlap_mask==1)}")
                valid_mask = sample_mask
                valid_sampled_indices = torch.where(valid_mask.bool()==1)
                # print(f"valid: {valid_sampled_indices}")
                #
                # valid_mask = torch.zeros(L, dtype=torch.bool, device=traj_emb.device)
                # valid_mask[valid_sampled_indices] = True

                """test valid mask"""
                # print(f"{len(valid_sampled_indices[0])} valid samples {valid_sampled_indices} in traj. of length {torch.sum(~src_padding_mask[b, :])} in pm {src_padding_mask.shape[-1]}" )
                # print()
                # assert torch.max(valid_sampled_indices[0]) < torch.sum(~src_padding_mask[b, :])

                curr_emb = cell_emb[b, :, :] * valid_mask.unsqueeze(-1)
                # curr_traj_o = traj_o[b, :, :] * valid_mask.unsqueeze(-1)
                curr_adj = adj[b, :, :, :] * valid_mask.unsqueeze(-1).unsqueeze(-1)
                # print(curr_t.shape)
                # print(curr_emb.shape)
                assert curr_emb.shape[0] == L
                # curr_emb = traj_emb[valid_sampled_indices, b, :]
                # curr_emb_p = traj_emb_p[valid_sampled_indices, b, :]

                aug_sampled_emb[m, b, :, :] = curr_emb
                # aug_sampled_traj_o[m, b, :, :] = curr_traj_o
                aug_sampled_adj[m, b, :] = curr_adj
                aug_src_padding_mask[m, b, :] = curr_src_pad

        # the final embeddings will be in [M, L, B, d].
        # print(f"creating {self.M*B} training samples.")

        return aug_sampled_emb, None, aug_src_padding_mask, aug_sampled_adj

    def loss_fn(self, context_out, targets):
        assert context_out.shape == targets.shape, "context_out and targets must have the same shape"
        M, B, L, d = context_out.shape

        # Calculate MSE loss for each element
        # mse_loss = F.mse_loss(context_out, targets, reduction='none')
        loss = self.loss(context_out, targets)

        loss = loss.sum(dim=(-1,-2))
        loss = loss.mean()

        return loss

    def interpret(self, cell_emb, traj_o, num_points, adj):
        max_traj_len = num_points.max().item()  # in essense -- trajs1_len[0]
        src_padding_mask = torch.arange(max_traj_len, device=Config.device)[None, :] >= num_points[:, None]
        traj_embs = self.context_encoder(**{
                'cell_emb': cell_emb,
                'traj_o': traj_o,
                'src_padding_mask': src_padding_mask,
                'adj': adj
            })
        return traj_embs

    def forward(self, cell_emb, traj_o, num_points, adj, mode):
        # create paddings for the varying-length input.
        # True indicates a padded position, and False indicates a valid data point
        # torch.Size([61, 128, 256]) torch.Size([61, 128, 4]) torch.Size([128, 61])
        # first is cell embedding dim., then point embedding dim, then padding mask dim.
        B, L, d = cell_emb.shape
        max_traj_len = num_points.max().item()  # in essense -- trajs1_len[0]
        src_padding_mask = torch.arange(max_traj_len, device=Config.device)[None, :] >= num_points[:, None]

        # if not training, encode the whole sequence.
        if mode != "train":
            return self.context_encoder(**{
                'cell_emb': cell_emb,
                'traj_o': traj_o,
                'src_padding_mask': src_padding_mask,
                'adj': adj
            })
        # ----- else -----
        # get target and context segments.
        target_emb, all_masks = self.target_seg((cell_emb, traj_o, num_points, src_padding_mask, adj), Config)

        aug_sampled_emb, aug_sampled_traj_o, aug_src_padding_mask, aug_sampled_adj = self.context_seg(
            (cell_emb, traj_o, src_padding_mask, adj), Config, all_masks)

        context_out = torch.zeros((self.M, B, L, d), device=cell_emb.device)
        # all_cls_token = torch.zeros((self.M, B, d), device=cell_emb.device)
        for m in range(self.M):
            # encoder
            # resulting target [L, B, d].
            out = self.context_encoder(**{
                'cell_emb': aug_sampled_emb[m],
                # 'traj_o': aug_sampled_traj_o[m],
                'traj_o': None,
                'src_padding_mask': aug_src_padding_mask[m],
                'adj': aug_sampled_adj[m]
            })

            # decoder
            # todo: will there be a valid memory mask?
            # tgt, memory, tgt_mask = target_emb, out, all_masks
            # tgt += self.pe
            # add latent variable z.
            mask = all_masks[m, :, :]  # [B, L]
            pe = self.pe.repeat(B, 1, 1)
            pe = pe[:, :L, :]
            # just get the PE.
            # pe = self.pe(out)
            pe = pe * mask.unsqueeze(-1).to(pe.dtype)
            z = self.latent_var.repeat(B, L, 1)
            z += pe
            # print(z.shape)
            out = torch.cat([out, z], dim=1)
            # cls_token = out[:, 0, :]
            # out = out.permute(1, 0, 2)
            # decode
            out = self.predictor(out)
            # for layer in self.layers:
            #     out = layer(out)

            # out = out.permute(1, 0, 2)
            out = out[:, L:, :]
            out = self.norm(out)
            # print(out.shape)

            out = self.proj_out(out)
            # print(out)
            # print(out.shape)
            # mask the non-target values according to all_masks.
            # out = out * mask.unsqueeze(-1).to(out.dtype)      # [B, L, d]
            context_out[m, :, :, :] = out
            # all_cls_token[m, :, :] = cls_token

        return context_out, target_emb

    def load_checkpoint(self):
        checkpoint_file = '{}/{}_Traj-JEPA_adj_fuse_new{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix,
                                                          Config.dumpfile_uniqueid)
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        return self


def test_masking():
    # jepa = JEPA_base()

    def mask_tensor(x, mask_ratio, succ_prob=0.5):
        """
        Mask a tensor with a given mask ratio, considering the padding mask and
        a probability to apply successive masking.

        Args:
        - traj_emb (torch.Tensor): The input tensor with shape [len, batch, d].
        - traj_emb_p (torch.Tensor): Positional embeddings tensor with shape [len, batch, 4].
        - src_padding_mask (torch.Tensor): Padding mask with shape [batch, len], True for padded positions.
        - mask_ratio (float): The ratio of len to be masked.
        - succ_prob (float): The probability of applying successive masking.

        Returns:
        - torch.Tensor: The masked tensor of traj_emb.
        """
        traj_emb, traj_emb_p, src_padding_mask = x
        L, B, _ = traj_emb.shape
        # Initialize a mask for all positions as True (indicating inclusion)
        mask = torch.ones((L, B), dtype=torch.bool, device=traj_emb.device)

        for b in range(B):
            # calculate the non-padding part.
            non_padded_len = L - src_padding_mask[b].sum().item()
            num_mask = int(non_padded_len * mask_ratio)
            print(f"masking {num_mask} points.")
            # Ensure num_mask is not zero to proceed with masking
            if num_mask > 0:
                if torch.rand(1).item() < succ_prob:
                    print("successive masking.")
                    # Successive masking
                    start = torch.randint(0, non_padded_len - num_mask + 1, (1,)).item()
                    mask[start:start + num_mask, b] = False
                else:
                    print("random masking.")
                    # Random masking
                    non_padded_indices = torch.nonzero(~src_padding_mask[b], as_tuple=False).squeeze()
                    mask_indices = non_padded_indices[torch.randperm(len(non_padded_indices))[:num_mask]]
                    mask[mask_indices, b] = False

        # Invert the source padding mask and overlay the target mask.
        # so that the masked parts are False.
        inv_src_padding_mask = ~src_padding_mask.transpose(0, 1)
        combined_mask = mask & inv_src_padding_mask
        print(combined_mask)
        # Apply the combined mask to traj_emb
        masked_traj_emb = traj_emb * combined_mask.unsqueeze(-1).to(traj_emb.dtype)
        masked_traj_emb_p = traj_emb_p * combined_mask.unsqueeze(-1).to(traj_emb_p.dtype)
        # print(masked_traj_emb[:, 0, :].shape)
        return masked_traj_emb, masked_traj_emb_p, src_padding_mask


    # seq1 = torch.randn(16, 2)  # Length 5
    # seq2 = torch.randn(10, 2)  # Length 3
    # seq3 = torch.randn(12, 2)  # Length 4
    #
    # # Batch size is 1 for this example, d is 2
    # batch_size = 1
    # d = 2
    #
    # # Manually pad sequences (for demonstration)
    # padded_seqs = pad_sequence([seq1, seq2, seq3], batch_first=False, padding_value=0)
    #
    # # Generate src_padding_mask
    # seq_lengths = torch.tensor([16, 10, 12])  # Actual lengths of sequences
    # max_length = padded_seqs.shape[0]
    # src_padding_mask = torch.arange(max_length)[None, :] >= seq_lengths[:, None]
    #
    # # Additional features tensor (dummy for demonstration)
    # traj_emb_p = torch.randn(max_length, batch_size, 4)
    #
    # # Mask ratio and successive probability
    # mask_ratio = 0.2
    # succ_prob = 0.5
    #
    # masked_traj_emb, masked_traj_emb_p, _ = mask_tensor((padded_seqs, traj_emb_p, src_padding_mask),
    #                                                     mask_ratio, succ_prob)

    # Base latitude and longitude for the starting point (somewhere in Manhattan)
    base_lat, base_lon = 40.7580, -73.9855  # Near Times Square

    # Generate trajectories with linear movement simulating blocks traveled
    # Trajectory 1: Length 18
    traj1 = torch.tensor([[base_lat + i * 0.001, base_lon + i * 0.002] for i in range(18)])
    # Trajectory 2: Length 12
    traj2 = torch.tensor([[base_lat + i * 0.0015, base_lon + i * 0.001] for i in range(12)])
    # Trajectory 3: Length 14
    traj3 = torch.tensor([[base_lat + i * 0.0005, base_lon - i * 0.0015] for i in range(14)])
    # Combine trajectories into a batch
    trajectories = [traj1, traj2, traj3]

    # Pad sequences to the same length
    padded_trajectories = pad_sequence(trajectories, batch_first=False, padding_value=0)  # Padding value 0 is arbitrary here
    batch_size = 1
    # d = 2
    # Determine sequence lengths before padding
    seq_lengths = torch.tensor([18, 12, 14])
    # seq_lengths = torch.tensor([len(traj) for traj in trajectories])

    # Generate the padding mask (True for padded positions)
    max_length = padded_trajectories.shape[0]  # Length after padding
    src_padding_mask = torch.arange(max_length)[None, :] >= seq_lengths[:, None]

    mask_ratio = 0.2
    succ_prob = 0.5

    masked_traj_emb, masked_traj_emb_p, _ = mask_tensor((padded_trajectories, padded_trajectories, src_padding_mask),
                                                        mask_ratio, succ_prob)
    # print(masked_traj_emb)
    np.save("vis_traj.npy", masked_traj_emb[:, 0, :].detach().numpy())
    np.save("ori_traj.npy", traj1)

"""
these are the inputs of TrajCL.
"""
 # trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len

 # we might need traj_emb, traj_emb_p, traj_len

if __name__ == '__main__':
    pass