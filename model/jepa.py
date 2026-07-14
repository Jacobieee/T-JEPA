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

        self.target_encoder = BaseBERT(Config, input_dim=2,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)

        self.context_encoder = BaseBERT(Config, input_dim=2,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)

        self.norm = nn.LayerNorm(Config.seq_embedding_dim)
        self.pe = nn.Parameter(torch.randn(1, Config.max_traj_len, Config.cell_embedding_dim))
        self.latent_var = nn.Parameter(torch.randn(1, Config.max_traj_len+1, Config.cell_embedding_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=Config.cell_embedding_dim,
            num_heads=4,
            batch_first=True
        )
        self.proj_q = nn.Linear(Config.cell_embedding_dim * 2, Config.cell_embedding_dim)

        # Normalize features before Linear projection
        self.traj_o_norm = nn.LayerNorm(2)  # Normalize the 2 input features (turning_angle, sinuosity)
        self.traj_o_proj = nn.Linear(2, Config.seq_embedding_dim)
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
    def mask_tensor(x, src_padding_mask, mask_ratio, succ_prob=0.5, m_count=4, valid=None, non_padded_len=None):
        """
        Vectorized generator for M masking groups.

        Args:
        - x: [B, L, d]
        - src_padding_mask: [B, L], True for padding positions
        - mask_ratio: float or scalar-like tensor
        - succ_prob: probability to use successive (contiguous) masking
        - m_count: number of masking groups M
        - valid: [B, L] (optional) precomputed valid mask
        - non_padded_len: [B] (optional) precomputed non-padded lengths

        Returns:
        - masked_x: [M, B, L, d]
        - masks: [M, B, L]
        - mask_indices: [] (placeholder for API compatibility)
        """
        B, L, _ = x.shape
        device = x.device

        # Valid positions (non-padding) - use precomputed if available
        if valid is None:
            valid = ~src_padding_mask  # [B, L]
        if non_padded_len is None:
            non_padded_len = valid.sum(dim=1)  # [B]
        num_mask = (non_padded_len.float() * float(mask_ratio)).long().clamp(min=0, max=L)  # [B]

        # Choose masking strategy per (m, b): successive vs random
        successive_choice = (torch.rand(m_count, B, device=device) < succ_prob) & (num_mask > 0).unsqueeze(0)

        # Random masking: select top-k per row using per-position random scores
        rand_scores = torch.rand(m_count, B, L, device=device)
        rand_scores = rand_scores.masked_fill(~valid.unsqueeze(0), -1.0)  # exclude padding by pushing them to the end


        num_mask_max = num_mask.max().item()
        if num_mask_max > 0:
            # Use topk to get top indices, k needs to be at least as large as max num_mask
            k = min(num_mask_max, L)
            _, top_indices = torch.topk(rand_scores, k=k, dim=2, largest=True)  # [M, B, k]
            
            # Initialize mask
            mask_random = torch.zeros(m_count, B, L, dtype=torch.bool, device=device)
            
            # For each (m, b), set top num_mask[b] positions to True
            # Use advanced indexing for vectorized assignment
            k_range = torch.arange(k, device=device).view(1, 1, k).expand(m_count, B, -1)  # [M, B, k]
            num_mask_expanded = num_mask.view(1, B, 1).expand(m_count, -1, k)  # [M, B, k]
            select_mask = k_range < num_mask_expanded  # [M, B, k] - True for positions to keep
            
            # Vectorized assignment using advanced indexing
            m_indices = torch.arange(m_count, device=device).view(m_count, 1, 1).expand(-1, B, k)
            b_indices = torch.arange(B, device=device).view(1, B, 1).expand(m_count, -1, k)
            mask_random[m_indices, b_indices, top_indices] = select_mask
            
            mask_random = mask_random & valid.view(1, B, L)  # [M, B, L]
        else:
            mask_random = torch.zeros(m_count, B, L, dtype=torch.bool, device=device)

        # Successive masking: sample a start index per (m, b), take a length of num_mask[b]
        span_max_start = (non_padded_len - num_mask).clamp_min(0)  # [B]
        start = (torch.rand(m_count, B, device=device) * (span_max_start + 1).float().unsqueeze(0)).floor().long()
        positions = torch.arange(L, device=device).view(1, 1, L)  # broadcasted index grid
        in_span = (positions >= start.unsqueeze(-1)) & (positions < (start + num_mask.view(1, B)).unsqueeze(-1))
        mask_successive = in_span & valid.view(1, B, L)  # [M, B, L]

        # Per (m, b) select which mask to use
        masks = torch.where(successive_choice.unsqueeze(-1), mask_successive, mask_random)  # [M, B, L]
        masked_x = x.unsqueeze(0) * masks.unsqueeze(-1).to(x.dtype)  # [M, B, L, d]

        return masked_x, masks, []


    @torch.no_grad()
    def target_seg(self, x, Config, valid=None, non_padded_len=None):
        """
        All we need is the target input, and the M * B masked segments.
        
        Args:
        - x: tuple containing (cell_emb, traj_o, num_points, src_padding_mask, adj)
        - Config: configuration object
        - valid: [B, L] (optional) precomputed valid mask
        - non_padded_len: [B] (optional) precomputed non-padded lengths
        """
        cell_emb, traj_o, num_points, src_padding_mask, adj = x
        # generate embeddings by target encoder.
        x_emb = self.target_encoder(**{
            'cell_emb': cell_emb,
            'traj_o': traj_o,
            'src_padding_mask': src_padding_mask,
            'adj': adj
        })  # [B, L, d]

        mask_ratio = Config.mask_ratio
        # get a random mask ratio and determine if the mask will be successive.
        idx = torch.randint(low=0, high=len(mask_ratio), size=(1,))
        r = mask_ratio[idx]

        all_mask_x, all_masks, _ = self.mask_tensor(x_emb, src_padding_mask, r, Config.succ_prob, self.M, valid, non_padded_len)


        return all_mask_x, all_masks


    def context_seg(self, x, Config, all_masks, valid=None, non_padded_len=None):
        cell_emb, traj_o, src_padding_mask, adj = x
        lb = Config.context_lb  # Lower bound ratio for sampling

        # Determine the number of sequence elements to sample based on the lower bound
        B, L, d = cell_emb.shape
        # Generate a unique sampling ratio for each trajectory in the batch within [lb, 1]
        ratios = torch.rand((B,), device=cell_emb.device).clamp(min=lb, max=1.0)
        # determine the number of samples kept considering the padding masks.
        # Use precomputed non_padded_len if available
        if non_padded_len is None:
            non_padded_lens = L - src_padding_mask.sum(dim=1)
        else:
            non_padded_lens = non_padded_len
        num_samples = (non_padded_lens.float() * ratios).long()

        # Vectorized context sampling: sample positions per (M, B) independently, then remove overlaps with all_masks
        # Use precomputed valid if available
        if valid is None:
            valid = ~src_padding_mask  # [B, L]
        
        # Sample positions per (M, B) using vectorized ranking (similar to mask_tensor)
        # Generate independent random scores for each M and B
        sample_scores = torch.rand(self.M, B, L, device=cell_emb.device)  # [M, B, L]
        sample_scores = sample_scores.masked_fill(~valid.unsqueeze(0), -1.0)  # exclude padding


        # Optimized version using topk instead of argsort
        num_samples_max = num_samples.max().item()
        if num_samples_max > 0:
            # Use topk to get top indices, k needs to be at least as large as max num_samples
            k = min(num_samples_max, L)
            _, top_indices = torch.topk(sample_scores, k=k, dim=2, largest=True)  # [M, B, k]
            
            # Initialize sample_mask
            sample_mask = torch.zeros(self.M, B, L, dtype=torch.bool, device=cell_emb.device)
            
            # For each (m, b), set top num_samples[b] positions to True
            # Use advanced indexing for vectorized assignment
            k_range = torch.arange(k, device=cell_emb.device).view(1, 1, k).expand(self.M, B, -1)  # [M, B, k]
            num_samples_expanded = num_samples.view(1, B, 1).expand(self.M, -1, k)  # [M, B, k]
            select_mask = k_range < num_samples_expanded  # [M, B, k] - True for positions to keep
            
            # Vectorized assignment using advanced indexing
            m_indices = torch.arange(self.M, device=cell_emb.device).view(self.M, 1, 1).expand(-1, B, k)
            b_indices = torch.arange(B, device=cell_emb.device).view(1, B, 1).expand(self.M, -1, k)
            sample_mask[m_indices, b_indices, top_indices] = select_mask
            
            sample_mask = sample_mask & valid.view(1, B, L)  # [M, B, L]
        else:
            sample_mask = torch.zeros(self.M, B, L, dtype=torch.bool, device=cell_emb.device)
        
        # Compute overlaps with all_masks [M, B, L]
        overlap_mask = sample_mask & all_masks  # [M, B, L]
        
        # Remove overlaps: final mask = sampled positions minus overlapping positions
        valid_mask = sample_mask & ~overlap_mask  # [M, B, L]
        
        # Apply masks to embeddings and adjacency
        aug_sampled_emb = cell_emb.unsqueeze(0) * valid_mask.unsqueeze(-1).to(cell_emb.dtype)  # [M, B, L, d]
        # aug_sampled_traj_o = traj_o.unsqueeze(0) * valid_mask.unsqueeze(-1).to(traj_o.dtype)  # [M, B, L, d]
        aug_sampled_traj_o = traj_o.unsqueeze(0).repeat(self.M, 1, 1, 1).to(traj_o.dtype)  # [M, B, L, d]
        aug_sampled_adj = adj.unsqueeze(0) * valid_mask.unsqueeze(-1).unsqueeze(-1).to(adj.dtype)  # [M, B, L, 9, d]
        aug_src_padding_mask = src_padding_mask.unsqueeze(0).expand(self.M, -1, -1)  # [M, B, L]

        # the final embeddings will be in [M, L, B, d].
        # print(f"creating {self.M*B} training samples.")

        return aug_sampled_emb, aug_sampled_traj_o, aug_src_padding_mask, aug_sampled_adj

    def loss_fn(self, context_out, targets):
        assert context_out.shape == targets.shape, "context_out and targets must have the same shape"
        M, B, L, d = context_out.shape

        # Calculate MSE loss for each element
        # mse_loss = F.mse_loss(context_out, targets, reduction='none')
        loss = self.loss(context_out, targets)

        loss = loss.sum(dim=-1)
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

        # Precompute valid and non_padded_len to avoid repeated calculation
        valid = ~src_padding_mask  # [B, L]
        non_padded_len = valid.sum(dim=1)  # [B]

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
        # Pass precomputed values to avoid recalculation
        target_emb, all_masks = self.target_seg((cell_emb, traj_o, num_points, src_padding_mask, adj), Config, valid, non_padded_len)

        aug_sampled_emb, aug_sampled_traj_o, aug_src_padding_mask, aug_sampled_adj = self.context_seg(
            (cell_emb, traj_o, src_padding_mask, adj), Config, all_masks, valid, non_padded_len)

        # Vectorized processing: merge M and B dimensions for batch processing
        # Reshape [M, B, ...] to [M*B, ...] for batch processing
        # Use reshape() instead of view() to handle potentially non-contiguous tensors (e.g., from expand())
        MB = self.M * B
        aug_sampled_emb_flat = aug_sampled_emb.reshape(MB, L, d)  # [M*B, L, d]
        aug_sampled_traj_o_flat = aug_sampled_traj_o.reshape(MB, L, 2)  # [M*B, L, 2]
        aug_src_padding_mask_flat = aug_src_padding_mask.reshape(MB, L)  # [M*B, L]
        aug_sampled_adj_flat = aug_sampled_adj.reshape(MB, L, Config.n_neighbors, d)  # [M*B, L, n_neighbors, d]
        all_masks_flat = all_masks.reshape(MB, L)  # [M*B, L]
        
        # Batch encoder: process all M*B samples at once
        out = self.context_encoder(**{
            'cell_emb': aug_sampled_emb_flat,
            'traj_o': None,
            'src_padding_mask': aug_src_padding_mask_flat,
            'adj': aug_sampled_adj_flat
        })  # [M*B, L, d]
        
        # Batch positional encoding and latent variable
        # pe = self.pe.repeat(MB, 1, 1)  # [M*B, max_len, d]
        # pe = pe[:, :L, :]  # [M*B, L, d]
        pe = self.pe.expand(MB, -1, -1)[:, :L, :]
        pe = pe * all_masks_flat.unsqueeze(-1).to(pe.dtype)  # [M*B, L, d]

        # prepare the latent variable.
        # z = self.latent_var.repeat(MB, 1, 1)  # [M*B, L, d]
        z = self.latent_var.expand(MB, -1, -1)[:, :L, :]
        # z = z[:, :L, :]  # [M*B, L, d]
        z = z + pe

        if mode == "train":
            _noise_scale = 0.001
            noise = torch.randn_like(z) * _noise_scale
            z = z + noise

        # Normalize features before Linear projection
        traj_o_norm = self.traj_o_norm(aug_sampled_traj_o_flat)  # [M*B, L, 2]
        feat_o = self.traj_o_proj(traj_o_norm)   # [M*B, L, d]

        z = torch.cat([z, feat_o], dim=-1)  # [M*B, L, 2*d]
        z = self.proj_q(z)  # [M*B, L, d]


        out, _ = self.cross_attn(z, out, out)
        out = out + z

        # Batch decoder
        out = self.predictor(out)  # [M*B, 2*L, d]
        
        # Extract prediction part and apply normalization and projection
        out = self.norm(out)  # [M*B, L, d]
        out = self.proj_out(out)  # [M*B, L, d]
        
        # Reshape back to [M, B, L, d]
        context_out = out.reshape(self.M, B, L, d)  # [M, B, L, d]

        return context_out, target_emb

    def interpret_ft(self, cell_emb, traj_o, num_points, adj):
        B, L, d = cell_emb.shape
        max_traj_len = num_points.max().item()  # in essense -- trajs1_len[0]
        src_padding_mask = torch.arange(max_traj_len, device=Config.device)[None, :] >= num_points[:, None]
        traj_embs = self.context_encoder(**{
                'cell_emb': cell_emb,
                'traj_o': traj_o,
                'src_padding_mask': src_padding_mask,
                'adj': adj
            })

        pe = self.pe.repeat(B, 1, 1)
        pe = pe[:, :L, :]

        z = self.latent_var.repeat(B, L, 1)
        z += pe


        return traj_embs, src_padding_mask

    def load_checkpoint(self):
        checkpoint_file = '{}/{}_t-jepa_porto_pretrain_motion_ca_noise_ep20_1e-4_mask234{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix,
                                                          Config.dumpfile_uniqueid)
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        return self

