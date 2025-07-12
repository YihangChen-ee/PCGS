#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
import torch.nn.functional as nnf
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized, Entropy_gaussian_mix_prob_2
from utils.entropy_models import Low_bound

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, \
    get_binary_vxl_size

from utils.encodings_cuda import \
    encoder, decoder, \
    encoder_gaussian_chunk, decoder_gaussian_chunk, \
    encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk, \
    encoder_trinomial, decoder_trinomial
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order

bit2MB_scale = 8 * 1024 * 1024
MAX_batch_size = 3000

def get_time():
    torch.cuda.synchronize()
    tt = time.time()
    return tt

class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]
        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        return out_i


class Channel_CTX_fea(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP_d0 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 0, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 1, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 2, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 3, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 4, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj


class Channel_CTX_fea_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10 * 1, 10 * 3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10 * 3, 10 * 3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(10 * 2, 10 * 3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10 * 3, 10 * 3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(10 * 3, 10 * 3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10 * 3, 10 * 3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(10 * 4, 10 * 3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10 * 3, 10 * 3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        NN = fea_q.shape[0]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj

class hash_ctx_pcgs(nn.Module):
    def __init__(self,
                 input_dim,
                 feat_dim,
                 output_dim,
                 lmbda_list_len=1,
                 lmbda_list=(1, 2, 4),
                 ):
        super().__init__()
        self.current_lmbda = 1e-3
        self.lmbda_list_len = lmbda_list_len
        self.cli = 0
        self.lmbda_list = lmbda_list
        self.l1 = nn.Linear(input_dim, feat_dim * 2)
        self.l2 = nn.Linear(feat_dim * 2, output_dim)
        self.l3 = nn.Linear(1, feat_dim * 2)
        self.hf_modulate_list = nn.Parameter(torch.rand(size=[lmbda_list_len, 2]), requires_grad=True)

        self.prob_feat_list = nn.Parameter(torch.rand(size=[lmbda_list_len, 2]), requires_grad=True)
        self.prob_feat_l1 = nn.Linear(input_dim + feat_dim, feat_dim * 3)
        self.prob_feat_l2 = nn.Linear(feat_dim * 3, feat_dim * 3)
        self.prob_feat_l3 = nn.Linear(1, feat_dim * 3)

        self.prob_scaling_list = nn.Parameter(torch.rand(size=[lmbda_list_len, 2]), requires_grad=True)
        self.prob_scaling_l1 = nn.Linear(input_dim + 6, 6 * 3)
        self.prob_scaling_l2 = nn.Linear(6 * 3, 6 * 3)
        self.prob_scaling_l3 = nn.Linear(1, 6 * 3)

        self.mlp_base = nn.Sequential(  # for feat, scaling, offset, respectively
            nn.Linear(input_dim, feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim, 3),
        )
        self.mlp_base_Goffsets = nn.Sequential(  # for feat, scaling, offset, respectively
            nn.Linear(input_dim, feat_dim * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim * 2, 60),
        )
    def forward(self, context_orig, cli):
        assert cli >= 0
        k = self.hf_modulate_list[cli:cli + 1, 0:1]
        c = self.hf_modulate_list[cli:cli + 1, 1:2]
        current_lmbda = torch.tensor([[self.lmbda_list[cli]]], device="cuda").float()  # [1, 1]
        current_lmbda = self.l3(torch.exp(k * current_lmbda + c))  # [1, feat_dim*2]

        f1 = self.l1(context_orig)  # [N, feat_dim*2]
        f1 = torch.relu(f1 * (1 + current_lmbda))
        ctx_info_old = self.l2(f1)

        Q_base = self.mlp_base(context_orig)  # [N, 3]
        Q_base_feat, Q_base_scaling, Q_base_offsets = torch.chunk(Q_base, 3, dim=-1)  # [N, 1] for each
        Q_base_feat = Q_base_feat.repeat(1, 50)
        Q_base_scaling = Q_base_scaling.repeat(1, 6)
        Q_base_offsets = Q_base_offsets.repeat(1, 30)

        G_base_offsets = self.mlp_base_Goffsets(context_orig)  # [N, 60]
        mean_offsets, scale_offsets = torch.chunk(G_base_offsets, chunks=2, dim=-1)  # [N, 30] for each

        return ctx_info_old, [Q_base_feat, Q_base_scaling, Q_base_offsets], None, [mean_offsets, scale_offsets]

    def forward_prob_feat(self, context_orig, feat_orig, Q_feat, cli, the_mean):
        Q_feat_ref = Q_feat * 3
        feat_q_ref = STE_multistep.apply(feat_orig / Q_feat_ref, 1) * Q_feat_ref
        feat_q = STE_multistep.apply(feat_orig / Q_feat, 1) * Q_feat
        indices = (((feat_q.detach() - feat_q_ref.detach()) / Q_feat).round().long()) + 1
        indices = torch.clamp(indices, 0, 2)
        mask = nnf.one_hot(indices, num_classes=3).to(dtype=torch.float)  # [N, feat_dim, 3]

        k = self.prob_feat_list[cli:cli + 1, 0:1]
        c = self.prob_feat_list[cli:cli + 1, 1:2]
        current_lmbda = torch.tensor([[self.lmbda_list[cli]]], device="cuda").float()  # [1, 1]
        current_lmbda = self.prob_feat_l3(torch.exp(k * current_lmbda + c))  # [1, feat_dim*3]

        f1 = self.prob_feat_l1(torch.cat([context_orig, feat_q_ref], dim=-1))  # [N, feat_dim*3]
        f1 = torch.relu(f1 * (1 + current_lmbda))
        pred_3prob = self.prob_feat_l2(f1)  # [N, feat_dim*3]
        pred_3prob = pred_3prob.view(pred_3prob.shape[0], -1, 3)  # [N, feat_dim, 3]
        pred_3prob = torch.softmax(pred_3prob, dim=-1)  # [N, feat_dim, 3]
        assert (pred_3prob.shape == mask.shape)

        pred_prob = torch.sum(pred_3prob * mask, dim=-1)  # [N, feat_dim]

        return pred_prob

    def enc_prob_feat(self, context_orig, feat_q, feat_q_ref, Q_feat, cli, the_mean, dec=False):

        k = self.prob_feat_list[cli:cli + 1, 0:1]
        c = self.prob_feat_list[cli:cli + 1, 1:2]
        current_lmbda = torch.tensor([[self.lmbda_list[cli]]], device="cuda").float()  # [1, 1]
        current_lmbda = self.prob_feat_l3(torch.exp(k * current_lmbda + c))  # [1, feat_dim*3]

        f1 = self.prob_feat_l1(torch.cat([context_orig, feat_q_ref], dim=-1))  # [N, feat_dim*3]
        f1 = torch.relu(f1 * (1 + current_lmbda))
        pred_3prob = self.prob_feat_l2(f1)  # [N, feat_dim*3]
        pred_3prob = pred_3prob.view(pred_3prob.shape[0], -1, 3)  # [N, feat_dim, 3]
        pred_3prob = torch.softmax(pred_3prob, dim=-1)  # [N, feat_dim, 3]
        if dec:
            return pred_3prob
        indices = (((feat_q.detach() - feat_q_ref.detach()) / Q_feat).round().long()) + 1
        indices = torch.clamp(indices, 0, 2)
        mask = nnf.one_hot(indices, num_classes=3).to(dtype=torch.float)  # [N, feat_dim, 3]
        assert (pred_3prob.shape == mask.shape)

        return pred_3prob, mask


    def forward_prob_scaling(self, context_orig, scaling_orig, Q_scaling, cli, the_mean):
        Q_scaling_ref = Q_scaling * 3
        scaling_q_ref = STE_multistep.apply(scaling_orig / Q_scaling_ref, 1) * Q_scaling_ref
        scaling_q = STE_multistep.apply(scaling_orig / Q_scaling, 1) * Q_scaling
        indices = (((scaling_q.detach() - scaling_q_ref.detach()) / Q_scaling).round().long()) + 1  # 确保 indices 是整数
        indices = torch.clamp(indices, 0, 2)  # 去除可能的越界，由于精度的问题。
        mask = nnf.one_hot(indices, num_classes=3).to(dtype=torch.float)  # [N, 6, 3]

        k = self.prob_scaling_list[cli:cli + 1, 0:1]
        c = self.prob_scaling_list[cli:cli + 1, 1:2]
        current_lmbda = torch.tensor([[self.lmbda_list[cli]]], device="cuda").float()  # [1, 1]
        current_lmbda = self.prob_scaling_l3(torch.exp(k * current_lmbda + c))  # [1, 6*3]

        f1 = self.prob_scaling_l1(torch.cat([context_orig, scaling_q_ref], dim=-1))  # [N, 6*3]
        f1 = torch.relu(f1 * (1 + current_lmbda))
        pred_3prob = self.prob_scaling_l2(f1)  # [N, 6*3]
        pred_3prob = pred_3prob.view(pred_3prob.shape[0], -1, 3)  # [N, 6, 3]
        pred_3prob = torch.softmax(pred_3prob, dim=-1)  # [N, 6, 3]
        assert (pred_3prob.shape == mask.shape)

        pred_prob = torch.sum(pred_3prob * mask, dim=-1)  # [N, 6]

        return pred_prob

    def enc_prob_scaling(self, context_orig, scaling_q, scaling_q_ref, Q_scaling, cli, the_mean, dec=False):

        k = self.prob_scaling_list[cli:cli + 1, 0:1]
        c = self.prob_scaling_list[cli:cli + 1, 1:2]
        current_lmbda = torch.tensor([[self.lmbda_list[cli]]], device="cuda").float()  # [1, 1]
        current_lmbda = self.prob_scaling_l3(torch.exp(k * current_lmbda + c))  # [1, 6*3]

        f1 = self.prob_scaling_l1(torch.cat([context_orig, scaling_q_ref], dim=-1))  # [N, 6*3]
        f1 = torch.relu(f1 * (1 + current_lmbda))
        pred_3prob = self.prob_scaling_l2(f1)  # [N, 6*3]
        pred_3prob = pred_3prob.view(pred_3prob.shape[0], -1, 3)  # [N, 6, 3]
        pred_3prob = torch.softmax(pred_3prob, dim=-1)  # [N, 6, 3]

        if dec:
            return pred_3prob

        indices = (((scaling_q.detach() - scaling_q_ref.detach()) / Q_scaling).round().long()) + 1
        indices = torch.clamp(indices, 0, 2)
        mask = nnf.one_hot(indices, num_classes=3).to(dtype=torch.float)  # [N, 6, 3]

        assert (pred_3prob.shape == mask.shape)

        return pred_3prob, mask


class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int = 50,
                 n_offsets: int = 5,
                 voxel_size: float = 0.01,
                 update_depth: int = 3,
                 update_init_factor: int = 100,
                 update_hierachy_factor: int = 4,
                 use_feat_bank=False,
                 n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 log2_hashmap_size_2D: int = 17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool = True,
                 ste_multistep: bool = False,
                 add_noise: bool = False,
                 Q=1,
                 use_2D: bool = True,
                 decoded_version: bool = False,
                 is_synthetic_nerf: bool = False,
                 lmbda_list=(1, 2, 4),
                 lmbda_list_len: int = 1,
                 ):
        super().__init__()
        print('hash_params:', use_2D, n_features_per_level,
              log2_hashmap_size, resolutions_list,
              log2_hashmap_size_2D, resolutions_list_2D,
              ste_binary, ste_multistep, add_noise)

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        x_bound_max = torch.ones(size=[1, 3], device='cuda')

        self.register_buffer("x_bound_min", x_bound_min)
        self.register_buffer("x_bound_max", x_bound_max)

        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version
        self.lmbda_list_len = lmbda_list_len

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not ste_binary: encoding_MB *= 32
        print(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3 + 1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim + 3 + 1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim + 3 + 1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim + 3 + 1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_grid = hash_ctx_pcgs(input_dim=self.encoding_xyz.output_dim,
                                     feat_dim=feat_dim,
                                     output_dim=(feat_dim + 6 + 3 * self.n_offsets) * 2 + feat_dim + 1 + 1 + 1,
                                     lmbda_list_len=lmbda_list_len,
                                     lmbda_list=lmbda_list,
                                     ).cuda()

        if not is_synthetic_nerf:
            self.mlp_deform = Channel_CTX_fea().cuda()
        else:
            print('find synthetic nerf, use Channel_CTX_fea_tiny')
            self.mlp_deform = Channel_CTX_fea_tiny().cuda()

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=1).cuda()
        self.current_lmbda = 1e-3
        self.current_lmbda_idx = 0

    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n:
                mlp_size += p.numel() * digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()
        self.mlp_deform.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.encoding_xyz.train()
        self.mlp_grid.train()
        self.mlp_deform.train()

        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._mask,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._anchor,
         self._offset,
         self._mask,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0 * self.scaling_activation(self._scaling)

    def get_transmit_mask(self, cli=None):
        if cli is None:
            cli = self.current_lmbda_idx

        if self.decoded_version:
            return self._mask[:, :10, cli:cli+1]

        mask_orig = self._mask[:, :10, 0:1]

        modify_v = torch.zeros_like(mask_orig)
        for i in range(1, cli + 1):
            modify_v = modify_v + nnf.softplus((self._mask[:, :10, i:i + 1]))

        mask_sig = torch.sigmoid(mask_orig + modify_v)
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    # @property
    def get_mask_anchor(self, cli=None):
        mask = self.get_transmit_mask(cli=cli)  # [N, 10, 1]
        mask_rate = torch.mean(mask, dim=1)  # [N, 1]
        mask_anchor = ((mask_rate > 0.0).float() - mask_rate).detach() + mask_rate
        return mask_anchor  # [N, 1]

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor = torch.round(self._anchor / self.voxel_size) * self.voxel_size
        anchor = anchor.detach() + (self._anchor - self._anchor.detach())
        return anchor

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.register_buffer("x_bound_min", x_bound_min)
        self.register_buffer("x_bound_max", x_bound_max)
        print('anchor_bound_updated', self.x_bound_min, self.x_bound_max)

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]
        return features

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets, self.lmbda_list_len)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init,
                 "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init,
                 "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init,
                 "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.offset_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.offset_lr_delay_mult,
                                                       max_steps=training_args.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init * self.spatial_lr_scale,
                                                     lr_final=training_args.mask_lr_final * self.spatial_lr_scale,
                                                     lr_delay_mult=training_args.mask_lr_delay_mult,
                                                     max_steps=training_args.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                            lr_final=training_args.mlp_opacity_lr_final,
                                                            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                            max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                          lr_final=training_args.mlp_color_lr_final,
                                                          lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                          max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                                    lr_final=training_args.mlp_featurebank_lr_final,
                                                                    lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                                    max_steps=training_args.mlp_featurebank_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                             lr_final=training_args.encoding_xyz_lr_final,
                                                             lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                             max_steps=training_args.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                         lr_final=training_args.mlp_grid_lr_final,
                                                         lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                         max_steps=training_args.mlp_grid_lr_max_steps,
                                                         step_sub=10000 if self.ste_binary else 10000,
                                                         )

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                           lr_final=training_args.mlp_deform_lr_final,
                                                           lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                           max_steps=training_args.mlp_deform_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mask":
                lr = self.mask_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1] * self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1] * self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        N = anchor.shape[0]
        opacities = opacities[:N]
        rotation = rotation[:N]
        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                           np.asarray(plydata.elements[0]["y"]),
                           np.asarray(plydata.elements[0]["z"])), axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key=lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(
            torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(
            torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(
            torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group[
                'name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask,
                        anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group[
                'name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):  # 3
            # for self.update_depth=3, self.update_hierachy_factor=4: 2**0, 2**1, 2**2
            cur_threshold = threshold * ((self.update_hierachy_factor // 2) ** i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')],
                                           dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            # for self.update_depth=3, self.update_hierachy_factor=4: 4**0, 4**1, 4**2
            size_factor = self.update_init_factor // (self.update_hierachy_factor ** i)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True,
                                                                        dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i * chunk_size:(
                                                                                                                                i + 1) * chunk_size,
                                                                                         :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[
                    candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][
                    remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat(
                    [1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat(
                    [1, self.n_offsets, self.lmbda_list_len]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat(
                    [self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat(
                    [self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity * self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1)  # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum() > 0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path):
        mkdir_p(os.path.dirname(path))

        if self.use_feat_bank:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)
        else:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)

    def load_mlp_checkpoints(self, path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])

    def contract_to_unisphere(self,
                              x: torch.Tensor,
                              aabb: torch.Tensor,
                              ord: int = 2,
                              eps: float = 1e-6,
                              derivative: bool = False,
                              ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag ** 2 + 2 * x ** 2 * (
                    1 / mag ** 3 - (2 * mag - 1) / mag ** 4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name):

        bit_anchor = 0
        bit_hash = 0
        bit_masks = 0
        bit_MLP = 0
        bit_feat_ttl = [0 for _ in range(self.lmbda_list_len)]  # bit for feat in new anchors
        bit_feat_inc_ttl = [0 for _ in range(self.lmbda_list_len)]  # bit for feat in refined anchors
        bit_scaling_ttl = [0 for _ in range(self.lmbda_list_len)]  # bit for scaling in new anchors
        bit_scaling_inc_ttl = [0 for _ in range(self.lmbda_list_len)]  # bit for scaling in refined anchors
        bit_offsets_ttl = [0 for _ in range(self.lmbda_list_len)]  # bit for offsets
        enc_time_st_ttl = [0 for _ in range(self.lmbda_list_len)]  # time for all operations in each level
        enc_time_ed_ttl = [0 for _ in range(self.lmbda_list_len)]
        enc_time_inc_st_ttl = [0 for _ in range(self.lmbda_list_len)]  # time for refined operations in each level
        enc_time_inc_ed_ttl = [0 for _ in range(self.lmbda_list_len)]

        for ss in range(self.lmbda_list_len):
            enc_time_st_ttl[ss] = get_time()
            if 1:
                if ss == 0:  # encode common elements, and create canvas for attributes
                    head_time_st = get_time()
                    mask_anchor_full = self.get_mask_anchor(cli=self.lmbda_list_len-1).to(torch.bool)[:, 0]  # [N]
                    N_valid = int(mask_anchor_full.sum().item())

                    # encode hash
                    hash_b_name = os.path.join(pre_path_name, 'hash.b')
                    hash_embeddings = self.get_encoding_params()  # {-1, 1}
                    bit_hash += encoder(((hash_embeddings.view(-1) + 1) / 2), file_name=hash_b_name)

                    # encode MLP
                    bit_MLP += self.get_mlp_size()[0]

                    # encode masks
                    _mask_Gaussian_canvas_valid = torch.zeros(size=[N_valid, self.n_offsets, self.lmbda_list_len], device="cuda", dtype=torch.bool)
                    for s in range(self.lmbda_list_len):
                        _mask_Gaussian_canvas_valid[:, :, s:s + 1] = (
                            self.get_transmit_mask(cli=s))[mask_anchor_full]

                    # encode anchor
                    _anchor_valid = self.get_anchor[mask_anchor_full]

                    _anchor_int = torch.round(_anchor_valid / self.voxel_size)
                    sorted_indices = calculate_morton_order(_anchor_int)
                    _anchor_int = _anchor_int[sorted_indices]
                    npz_path = os.path.join(pre_path_name, 'xyz_gpcc.npz')
                    means_strings = compress_gpcc(_anchor_int)
                    np.savez_compressed(npz_path, voxel_size=self.voxel_size, means_strings=means_strings)
                    bit_anchor += os.path.getsize(npz_path) * 8

                    # obtain and reorder
                    _feat_valid = self._anchor_feat[mask_anchor_full]  # N, 50
                    _grid_offsets_valid = self._offset[mask_anchor_full]  # N, 10, 3
                    _scaling_valid = self.get_scaling[mask_anchor_full]  # N, 6

                    _anchor_valid = _anchor_int * self.voxel_size
                    _feat_valid = _feat_valid[sorted_indices]
                    _grid_offsets_valid = _grid_offsets_valid[sorted_indices]
                    _scaling_valid = _scaling_valid[sorted_indices]
                    _mask_Gaussian_canvas_valid = _mask_Gaussian_canvas_valid[sorted_indices]
                    _mask_anchor_canvas_valid = (torch.sum(_mask_Gaussian_canvas_valid, dim=1) > 0)  # [N_valid, lmd_len]

                    # mask_canvas_b_name = os.path.join(pre_path_name, 'mask_canvas.b')
                    bit_masks += encoder(_mask_Gaussian_canvas_valid[..., 0], file_name=os.path.join(pre_path_name, f'mask_canvas_ss{0}.b'))
                    for ii in range(1, self.lmbda_list_len):
                        tmp = _mask_Gaussian_canvas_valid[..., ii].float() - _mask_Gaussian_canvas_valid[..., ii-1].float()
                        bit_masks += encoder(tmp, file_name=os.path.join(pre_path_name, f'mask_canvas_ss{ii}.b'))

                    _mask_Gaussian_canvas_valid = _mask_Gaussian_canvas_valid.to(torch.bool)
                    _mask_anchor_canvas_valid = _mask_anchor_canvas_valid.to(torch.bool)

                    torch.save(self.x_bound_min, os.path.join(pre_path_name, 'x_bound_min.pkl'))
                    torch.save(self.x_bound_max, os.path.join(pre_path_name, 'x_bound_max.pkl'))

                    feat_canvas_valid = torch.zeros(size=[N_valid, self.feat_dim], device="cuda")
                    scaling_canvas_valid = torch.zeros(size=[N_valid, 6], device="cuda")
                    offsets_canvas_valid = torch.zeros(size=[N_valid, self.n_offsets, 3], device="cuda")

                    head_time_ed = get_time()

            if 1:
                if ss == 0:
                    mask_anchor_curr = _mask_anchor_canvas_valid[:, ss]
                else:
                    mask_anchor_curr = (_mask_anchor_canvas_valid[:, ss].float() - _mask_anchor_canvas_valid[:, ss-1].float()).to(torch.bool)
                _anchor_curr = _anchor_valid[mask_anchor_curr]
                _feat_curr = _feat_valid[mask_anchor_curr]
                _scaling_curr = _scaling_valid[mask_anchor_curr]

                N_curr = _anchor_curr.shape[0]
                steps = (N_curr // MAX_batch_size) if (N_curr % MAX_batch_size) == 0 else (N_curr // MAX_batch_size + 1)

                _feat_slice_q_list = []
                _scaling_slice_q_list = []

                for s in range(steps):
                    N_start = s * MAX_batch_size
                    N_end = min((s + 1) * MAX_batch_size, N_curr)

                    feat_b_name = os.path.join(pre_path_name, f'ss{ss}_feat.b').replace('.b', f'_{s}.b')
                    scaling_b_name = os.path.join(pre_path_name, f'ss{ss}_scaling.b').replace('.b', f'_{s}.b')

                    _anchor_slice = _anchor_curr[N_start:N_end]

                    # encode feat
                    feat_context_orig = self.calc_interp_feat(_anchor_slice)  # [N_num, ?]
                    feat_context_ttl = self.get_grid_mlp.forward(feat_context_orig, cli=ss)
                    ctx_info_old, Q_base = feat_context_ttl[0], feat_context_ttl[1]
                    mean, scale, prob, mean_scaling, scale_scaling, _, _, _, _, _ = \
                        torch.split(ctx_info_old, split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
                    Q_feat_adj, Q_scaling_adj = Q_base[0], Q_base[1]
                    Q_feat_basic = 1 * (1 + torch.tanh(Q_feat_adj.contiguous()))
                    Q_scaling_basic = 0.001 * (1 + torch.tanh(Q_scaling_adj.contiguous()))

                    Q_feat = Q_feat_basic / (3 ** ss)
                    Q_scaling = Q_scaling_basic / (3 ** ss)

                    Q_scaling = Q_scaling.view(-1)
                    mean_scaling = mean_scaling.contiguous().view(-1)
                    scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)

                    _feat_slice = _feat_curr[N_start:N_end]
                    _feat_slice_q = STE_multistep.apply(_feat_slice, Q_feat, self._anchor_feat.mean())
                    _feat_slice_q_list.append(_feat_slice_q.view(-1, self.feat_dim))

                    mean_scale = torch.cat([mean, scale, prob], dim=-1)
                    scale = torch.clamp(scale, min=1e-9)
                    for cc in range(5):
                        mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat_slice_q, mean_scale, to_dec=cc)
                        probs = torch.stack([prob[:, cc * 10:cc * 10 + 10], prob_adj], dim=-1)
                        probs = torch.softmax(probs, dim=-1)

                        feat_tmp = _feat_slice_q[:, cc * 10:cc * 10 + 10].contiguous().view(-1)
                        Q_feat_tmp = Q_feat[:, cc * 10:cc * 10 + 10].contiguous().view(-1)

                        bit_feat_ttl[ss] += encoder_gaussian_mixed_chunk(
                            feat_tmp,
                            [mean[:, cc * 10:cc * 10 + 10].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                            [scale[:, cc * 10:cc * 10 + 10].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                            [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                            Q_feat_tmp,
                            file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)

                    _scaling_slice = _scaling_curr[N_start:N_end].view(-1)  # [N_num*6]
                    _scaling_slice_q = STE_multistep.apply(_scaling_slice, Q_scaling, self.get_scaling.mean())
                    bit_scaling_ttl[ss] += encoder_gaussian_chunk(_scaling_slice_q, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
                    _scaling_slice_q_list.append(_scaling_slice_q.view(-1, 6))

                    torch.cuda.empty_cache()

                if len(_feat_slice_q_list) > 0:
                    feat_canvas_valid[mask_anchor_curr] = torch.cat(_feat_slice_q_list, dim=0)

                if len(_scaling_slice_q_list) > 0:
                    scaling_canvas_valid[mask_anchor_curr] = torch.cat(_scaling_slice_q_list, dim=0)

            if 1:
                enc_time_inc_st_ttl[ss] = get_time()
                if ss > 0:
                    mask_anchor_curr = (_mask_anchor_canvas_valid[:, ss].float() * _mask_anchor_canvas_valid[:, ss-1].float()).to(torch.bool)

                    _anchor_curr = _anchor_valid[mask_anchor_curr]
                    _feat_curr = _feat_valid[mask_anchor_curr]
                    _scaling_curr = _scaling_valid[mask_anchor_curr]

                    N_curr = _anchor_curr.shape[0]
                    steps = (N_curr // MAX_batch_size) if (N_curr % MAX_batch_size) == 0 else (N_curr // MAX_batch_size + 1)

                    _feat_slice_q_list = []
                    _scaling_slice_q_list = []

                    for s in range(steps):
                        N_start = s * MAX_batch_size
                        N_end = min((s + 1) * MAX_batch_size, N_curr)

                        feat_b_name = os.path.join(pre_path_name, f'ss{ss}_feat_inc.b').replace('.b', f'_{s}.b')
                        scaling_b_name = os.path.join(pre_path_name, f'ss{ss}_scaling_inc.b').replace('.b', f'_{s}.b')

                        _anchor_slice = _anchor_curr[N_start:N_end]

                        feat_context_orig = self.calc_interp_feat(_anchor_slice)  # [N_num, ?]
                        feat_context_ttl = self.get_grid_mlp.forward(feat_context_orig, cli=ss)
                        ctx_info_old, Q_base = feat_context_ttl[0], feat_context_ttl[1]

                        Q_feat_adj, Q_scaling_adj = Q_base[0], Q_base[1]
                        Q_feat_basic = 1 * (1 + torch.tanh(Q_feat_adj.contiguous()))
                        Q_scaling_basic = 0.001 * (1 + torch.tanh(Q_scaling_adj.contiguous()))

                        Q_feat = Q_feat_basic / (3 ** ss)
                        Q_scaling = Q_scaling_basic / (3 ** ss)

                        _feat_slice = _feat_curr[N_start:N_end]
                        _feat_slice_q = STE_multistep.apply(_feat_slice, Q_feat, self._anchor_feat.mean())
                        _feat_slice_q_ref = feat_canvas_valid[mask_anchor_curr][N_start:N_end]
                        prob3, mask = self.get_grid_mlp.enc_prob_feat(feat_context_orig, _feat_slice_q, _feat_slice_q_ref, Q_feat, ss, self._anchor_feat.mean())
                        indices = torch.argmax(mask, dim=-1)
                        _feat_slice_q = _feat_slice_q_ref + Q_feat * (indices - 1)

                        bit_feat_inc_ttl[ss] += encoder_trinomial(mask, prob3, file_name=feat_b_name)
                        _feat_slice_q_list.append(_feat_slice_q.view(-1, self.feat_dim))

                        _scaling_slice = _scaling_curr[N_start:N_end]
                        _scaling_slice_q = STE_multistep.apply(_scaling_slice, Q_scaling, self.get_scaling.mean())
                        _scaling_slice_q_ref = scaling_canvas_valid[mask_anchor_curr][N_start:N_end]
                        prob3, mask = self.get_grid_mlp.enc_prob_scaling(feat_context_orig, _scaling_slice_q, _scaling_slice_q_ref, Q_scaling, ss, self.get_scaling.mean())
                        indices = torch.argmax(mask, dim=-1)
                        _scaling_slice_q = _scaling_slice_q_ref + Q_scaling * (indices - 1)

                        bit_scaling_inc_ttl[ss] += encoder_trinomial(mask, prob3, file_name=scaling_b_name)
                        _scaling_slice_q_list.append(_scaling_slice_q.view(-1, 6))

                    if len(_feat_slice_q_list) > 0:
                        feat_canvas_valid[mask_anchor_curr] = torch.cat(_feat_slice_q_list, dim=0)

                    if len(_scaling_slice_q_list) > 0:
                        scaling_canvas_valid[mask_anchor_curr] = torch.cat(_scaling_slice_q_list, dim=0)
                enc_time_inc_ed_ttl[ss] = get_time()

            if 1:
                if ss == 0:
                    mask_anchor_curr = _mask_anchor_canvas_valid[:, ss]
                    mask_Gaussian_curr = _mask_Gaussian_canvas_valid[mask_anchor_curr, :, ss:ss+1]  # [N_curr, 10, 1]
                else:
                    mask_anchor_curr = (torch.sum(_mask_Gaussian_canvas_valid[:, :, ss].float() - _mask_Gaussian_canvas_valid[:, :, ss-1].float(), dim=1) > 0).to(torch.bool)  # [N_valid, 10] -> [N_valid]
                    mask_Gaussian_curr = (_mask_Gaussian_canvas_valid[mask_anchor_curr, :, ss:ss+1].float() - _mask_Gaussian_canvas_valid[mask_anchor_curr, :, ss-1:ss].float()).to(torch.bool)  # [N_curr, 10, 1]

                _anchor_curr = _anchor_valid[mask_anchor_curr]  # [N_curr, 3]
                _grid_offsets_curr = _grid_offsets_valid[mask_anchor_curr]  # [N_curr, 10, 3]

                N_curr = _anchor_curr.shape[0]
                steps = (N_curr // MAX_batch_size) if (N_curr % MAX_batch_size) == 0 else (N_curr // MAX_batch_size + 1)

                zzz_list = []

                for s in range(steps):
                    N_start = s * MAX_batch_size
                    N_end = min((s + 1) * MAX_batch_size, N_curr)

                    offsets_b_name = os.path.join(pre_path_name, f'ss{ss}_offsets.b').replace('.b', f'_{s}.b')

                    _anchor_slice = _anchor_curr[N_start:N_end]

                    # encode feat
                    feat_context_orig = self.calc_interp_feat(_anchor_slice)  # [N_num, ?]
                    feat_context_ttl = self.get_grid_mlp.forward(feat_context_orig, cli=ss)
                    ctx_info_old, Q_base = feat_context_ttl[0], feat_context_ttl[1]

                    mean_offsets, scale_offsets = feat_context_ttl[3]
                    Q_offsets_adj = Q_base[2]

                    Q_offsets = 0.2 * (1 + torch.tanh(Q_offsets_adj.contiguous().view(-1, self.n_offsets, 3)))

                    Q_offsets = Q_offsets.view(-1)
                    mean_offsets = mean_offsets.contiguous().view(-1)
                    scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)

                    mask_slice = mask_Gaussian_curr[N_start:N_end]
                    mask_slice = mask_slice.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
                    offsets_slice = _grid_offsets_curr[N_start:N_end].view(-1, 3 * self.n_offsets).view(-1)  # [N_num*K*3]
                    offsets_slice[~mask_slice] = 0.0
                    offsets_slice_q = STE_multistep.apply(offsets_slice, Q_offsets, self._offset.mean())
                    offsets_slice_q[~mask_slice] = 0.0
                    bit_offsets_ttl[ss] += encoder_gaussian_chunk(offsets_slice_q[mask_slice], mean_offsets[mask_slice], scale_offsets[mask_slice], Q_offsets[mask_slice], file_name=offsets_b_name, chunk_size=3_0000)

                    zzz_list.append(offsets_slice_q.view(-1, 3 * self.n_offsets).view(-1, self.n_offsets, 3))

                zz2 = torch.cat(zzz_list, dim=0)

                offsets_canvas_valid[mask_anchor_curr] += zz2
            enc_time_ed_ttl[ss] = get_time()

        log_info_list = []

        for ss in range(self.lmbda_list_len):
            log_info = f"\n[ITER final Step {ss}] Encoded sizes in MB: " \
                       f"lmd_info {self.current_lmbda_idx}, {self.current_lmbda} || " \
                       f"ttl_anchor {self._anchor.shape[0]}, " \
                       f"ttl_Gaussian {self._anchor.shape[0]*self.n_offsets}, " \
                       f"1_rate_mask_anchor {round(self.get_mask_anchor(cli=ss).mean().item(), 6)}, " \
                       f"1_rate_mask_Gaussian {round(self.get_transmit_mask(cli=ss).mean().item(), 6)}, " \
                       f"hash {round(bit_hash / bit2MB_scale, 6) if ss==0 else 0}, " \
                       f"masks {round(bit_masks / bit2MB_scale, 6) if ss==0 else 0}, " \
                       f"anchor {round(bit_anchor / bit2MB_scale, 6) if ss==0 else 0}, " \
                       f"MLPs {round(bit_MLP / bit2MB_scale, 6) if ss==0 else 0}, " \
                       f"feat {round(bit_feat_ttl[ss] / bit2MB_scale, 6)}, " \
                       f"feat_inc {round(bit_feat_inc_ttl[ss] / bit2MB_scale, 6)}, " \
                       f"scaling {round(bit_scaling_ttl[ss] / bit2MB_scale, 6)}, " \
                       f"scaling_inc {round(bit_scaling_inc_ttl[ss] / bit2MB_scale, 6)}, " \
                       f"offsets {round(bit_offsets_ttl[ss] / bit2MB_scale, 6)}, " \
                       f"head_time {round(head_time_ed - head_time_st, 6) if ss==0 else 0}, " \
                       f"enc_inc_time {round(enc_time_inc_ed_ttl[ss] - enc_time_inc_st_ttl[ss], 6) if ss>0 else 0}, " \
                       f"ttl_time (w/head w/inc) {round(enc_time_ed_ttl[ss] - enc_time_st_ttl[ss], 6)}"
            log_info_list.append(log_info)

        return log_info_list

    @torch.no_grad()
    def conduct_decoding(self, pre_path_name, decode_until=4):

        print('Start decoding ...')

        dec_time_st_ttl = [0 for _ in range(self.lmbda_list_len)]
        dec_time_ed_ttl = [0 for _ in range(self.lmbda_list_len)]
        dec_time_inc_st_ttl = [0 for _ in range(self.lmbda_list_len)]
        dec_time_inc_ed_ttl = [0 for _ in range(self.lmbda_list_len)]

        for ss in range(decode_until):
            dec_time_st_ttl[ss] = get_time()
            if 1:
                if ss == 0:

                    head_time_st = get_time()

                    self.x_bound_min = torch.load(os.path.join(pre_path_name, 'x_bound_min.pkl'))
                    self.x_bound_max = torch.load(os.path.join(pre_path_name, 'x_bound_max.pkl'))

                    hash_b_name = os.path.join(pre_path_name, 'hash.b')
                    N_hash = torch.zeros_like(self.get_encoding_params()).numel()
                    hash_embeddings = decoder(N_hash, hash_b_name)  # {0, 1}
                    hash_embeddings = (hash_embeddings * 2 - 1).to(torch.float32)
                    hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)

                    npz_path = os.path.join(pre_path_name, 'xyz_gpcc.npz')
                    data_dict = np.load(npz_path)
                    voxel_size = float(data_dict['voxel_size'])
                    means_strings = data_dict['means_strings'].tobytes()
                    _anchor_int_dec = decompress_gpcc(means_strings).to('cuda')
                    sorted_indices = calculate_morton_order(_anchor_int_dec)
                    _anchor_int_dec = _anchor_int_dec[sorted_indices]
                    _anchor_valid = _anchor_int_dec * voxel_size

                    N_valid = _anchor_valid.shape[0]

                    # mask_canvas_b_name = os.path.join(pre_path_name, 'mask_canvas.b')
                    _mask_Gaussian_canvas_valid = torch.zeros(size=[N_valid, self.n_offsets, self.lmbda_list_len], device="cuda", dtype=torch.float)
                    tmp = decoder(N_valid * self.n_offsets, os.path.join(pre_path_name, f'mask_canvas_ss{0}.b'))
                    tmp = tmp.view(N_valid, self.n_offsets)
                    _mask_Gaussian_canvas_valid[..., 0] = tmp
                    for ii in range(1, self.lmbda_list_len):
                        tmp = decoder(N_valid * self.n_offsets, os.path.join(pre_path_name, f'mask_canvas_ss{ii}.b'))
                        tmp = tmp.view(N_valid, self.n_offsets)
                        _mask_Gaussian_canvas_valid[..., ii] = _mask_Gaussian_canvas_valid[..., ii-1] + tmp

                    _mask_anchor_canvas_valid = (torch.sum(_mask_Gaussian_canvas_valid, dim=1) > 0)  # [N_valid, lmd_len]

                    _mask_Gaussian_canvas_valid = _mask_Gaussian_canvas_valid.to(torch.bool)
                    _mask_anchor_canvas_valid = _mask_anchor_canvas_valid.to(torch.bool)

                    feat_canvas_valid = torch.zeros(size=[N_valid, self.feat_dim], device="cuda", dtype=torch.float)
                    scaling_canvas_valid = torch.zeros(size=[N_valid, 6], device="cuda", dtype=torch.float)
                    offsets_canvas_valid = torch.zeros(size=[N_valid, self.n_offsets, 3], device="cuda", dtype=torch.float)

                    head_time_ed = get_time()

            if 1:  # for newly appeared anchors
                if ss == 0:
                    mask_anchor_curr = _mask_anchor_canvas_valid[:, ss]  # all anchors in level 0
                else:
                    mask_anchor_curr = (_mask_anchor_canvas_valid[:, ss].float() - _mask_anchor_canvas_valid[:, ss-1].float()).to(torch.bool)  # new anchors in level s
                _anchor_curr = _anchor_valid[mask_anchor_curr]
                N_curr = _anchor_curr.shape[0]
                steps = (N_curr // MAX_batch_size) if (N_curr % MAX_batch_size) == 0 else (N_curr // MAX_batch_size + 1)

                _feat_slice_q_list = []
                _scaling_slice_q_list = []
                for s in range(steps):

                    N_start = s * MAX_batch_size
                    N_end = min((s + 1) * MAX_batch_size, N_curr)
                    N_num = N_end - N_start
                    # sizes of MLPs is not included here
                    feat_b_name = os.path.join(pre_path_name, f'ss{ss}_feat.b').replace('.b', f'_{s}.b')
                    scaling_b_name = os.path.join(pre_path_name, f'ss{ss}_scaling.b').replace('.b', f'_{s}.b')

                    # encode feat
                    _anchor_slice = _anchor_curr[N_start:N_end]
                    feat_context_orig = self.calc_interp_feat(_anchor_slice)  # [N_num, ?]
                    feat_context_ttl = self.get_grid_mlp.forward(feat_context_orig, cli=ss)
                    ctx_info_old, Q_base = feat_context_ttl[0], feat_context_ttl[1]
                    mean, scale, prob, mean_scaling, scale_scaling, _, _, _, _, _ = \
                        torch.split(ctx_info_old, split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
                    Q_feat_adj, Q_scaling_adj = Q_base[0], Q_base[1]
                    Q_feat_basic = 1 * (1 + torch.tanh(Q_feat_adj.contiguous()))
                    Q_scaling_basic = 0.001 * (1 + torch.tanh(Q_scaling_adj.contiguous()))

                    Q_feat = Q_feat_basic / (3 ** ss)
                    Q_scaling = Q_scaling_basic / (3 ** ss)

                    Q_scaling = Q_scaling.view(-1)
                    mean_scaling = mean_scaling.contiguous().view(-1)
                    scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)

                    _feat_slice_q = torch.zeros(size=[N_num, self.feat_dim], device='cuda', dtype=torch.float32)
                    mean_scale = torch.cat([mean, scale, prob], dim=-1)
                    scale = torch.clamp(scale, min=1e-9)
                    for cc in range(5):
                        mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat_slice_q, mean_scale, to_dec=cc)
                        probs = torch.stack([prob[:, cc * 10:cc * 10 + 10], prob_adj], dim=-1)
                        probs = torch.softmax(probs, dim=-1)
                        Q_feat_tmp = Q_feat[:, cc * 10:cc * 10 + 10].contiguous().view(-1)

                        feat_decoded_tmp = decoder_gaussian_mixed_chunk(
                            [mean[:, cc * 10:cc * 10 + 10].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                            [scale[:, cc * 10:cc * 10 + 10].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                            [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                            Q_feat_tmp,
                            file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)

                        feat_decoded_tmp = feat_decoded_tmp.view(N_num, 10)
                        _feat_slice_q[:, cc * 10:cc * 10 + 10] = feat_decoded_tmp
                    _feat_slice_q_list.append(_feat_slice_q)

                    _scaling_slice_q = decoder_gaussian_chunk(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
                    _scaling_slice_q = _scaling_slice_q.view(N_num, 6)  # [N_num, 6]
                    _scaling_slice_q_list.append(_scaling_slice_q)

                if len(_feat_slice_q_list) > 0:
                    _feat_curr = torch.cat(_feat_slice_q_list, dim=0)
                    feat_canvas_valid[mask_anchor_curr] = _feat_curr

                if len(_scaling_slice_q_list) > 0:
                    _scaling_curr = torch.cat(_scaling_slice_q_list, dim=0)
                    scaling_canvas_valid[mask_anchor_curr] = _scaling_curr

            if 1:
                dec_time_inc_st_ttl[ss] = get_time()
                if ss > 0:
                    mask_anchor_curr = (_mask_anchor_canvas_valid[:, ss].float() * _mask_anchor_canvas_valid[:, ss-1].float()).to(torch.bool)

                    _anchor_curr = _anchor_valid[mask_anchor_curr]

                    N_curr = _anchor_curr.shape[0]
                    steps = (N_curr // MAX_batch_size) if (N_curr % MAX_batch_size) == 0 else (N_curr // MAX_batch_size + 1)

                    feat_q_list = []
                    scaling_q_list = []

                    for s in range(steps):
                        N_start = s * MAX_batch_size
                        N_end = min((s + 1) * MAX_batch_size, N_curr)

                        feat_b_name = os.path.join(pre_path_name, f'ss{ss}_feat_inc.b').replace('.b', f'_{s}.b')
                        scaling_b_name = os.path.join(pre_path_name, f'ss{ss}_scaling_inc.b').replace('.b', f'_{s}.b')

                        _anchor_slice = _anchor_curr[N_start:N_end]

                        feat_context_orig = self.calc_interp_feat(_anchor_slice)  # [N_num, ?]
                        feat_context_ttl = self.get_grid_mlp.forward(feat_context_orig, cli=ss)
                        ctx_info_old, Q_base = feat_context_ttl[0], feat_context_ttl[1]

                        Q_feat_adj, Q_scaling_adj = Q_base[0], Q_base[1]
                        Q_feat_basic = 1 * (1 + torch.tanh(Q_feat_adj.contiguous()))
                        Q_scaling_basic = 0.001 * (1 + torch.tanh(Q_scaling_adj.contiguous()))

                        Q_feat = Q_feat_basic / (3 ** ss)
                        Q_scaling = Q_scaling_basic / (3 ** ss)

                        feat_q_ref = feat_canvas_valid[mask_anchor_curr][N_start:N_end]
                        prob3 = self.get_grid_mlp.enc_prob_feat(feat_context_orig, None, feat_q_ref, None, ss, self._anchor_feat.mean(), dec=True)
                        mask = decoder_trinomial(prob3, file_name=feat_b_name)
                        indices = torch.argmax(mask, dim=-1)
                        feat_q = feat_q_ref + Q_feat * (indices - 1)

                        feat_q_list.append(feat_q)

                        scaling_q_ref = scaling_canvas_valid[mask_anchor_curr][N_start:N_end]
                        prob3 = self.get_grid_mlp.enc_prob_scaling(feat_context_orig, None, scaling_q_ref, None, ss, self.get_scaling.mean(), dec=True)
                        mask = decoder_trinomial(prob3, file_name=scaling_b_name)
                        indices = torch.argmax(mask, dim=-1)
                        scaling_q = scaling_q_ref + Q_scaling * (indices - 1)

                        scaling_q_list.append(scaling_q)

                    if len(feat_q_list) > 0:
                        feat_q_curr = torch.cat(feat_q_list, dim=0)
                        feat_canvas_valid[mask_anchor_curr] = feat_q_curr

                    if len(scaling_q_list) > 0:
                        scaling_q_curr = torch.cat(scaling_q_list, dim=0)
                        scaling_canvas_valid[mask_anchor_curr] = scaling_q_curr

                dec_time_inc_ed_ttl[ss] = get_time()

            if 1:
                if ss == 0:
                    mask_anchor_curr = _mask_anchor_canvas_valid[:, ss]
                    mask_Gaussian_curr = _mask_Gaussian_canvas_valid[mask_anchor_curr, :, ss:ss+1]  # [N_curr, 10, 1]
                else:
                    mask_anchor_curr = (torch.sum(_mask_Gaussian_canvas_valid[:, :, ss].float() - _mask_Gaussian_canvas_valid[:, :, ss-1].float(), dim=1) > 0).to(torch.bool)  # [N_valid, 10] -> [N_valid]
                    mask_Gaussian_curr = (_mask_Gaussian_canvas_valid[mask_anchor_curr, :, ss:ss+1].float() - _mask_Gaussian_canvas_valid[mask_anchor_curr, :, ss-1:ss].float()).to(torch.bool)  # [N_curr, 10, 1]

                _anchor_curr = _anchor_valid[mask_anchor_curr]  # [N_curr, 3]

                N_curr = _anchor_curr.shape[0]
                steps = (N_curr // MAX_batch_size) if (N_curr % MAX_batch_size) == 0 else (N_curr // MAX_batch_size + 1)

                offsets_slice_q_list = []

                for s in range(steps):
                    N_num = min(MAX_batch_size, N_curr - s * MAX_batch_size)
                    N_start = s * MAX_batch_size
                    N_end = min((s + 1) * MAX_batch_size, N_curr)

                    offsets_b_name = os.path.join(pre_path_name, f'ss{ss}_offsets.b').replace('.b', f'_{s}.b')

                    _anchor_slice = _anchor_curr[N_start:N_end]

                    # encode feat
                    feat_context_orig = self.calc_interp_feat(_anchor_slice)  # [N_num, ?]
                    feat_context_ttl = self.get_grid_mlp.forward(feat_context_orig, cli=ss)
                    ctx_info_old, Q_base = feat_context_ttl[0], feat_context_ttl[1]

                    mean_offsets, scale_offsets = feat_context_ttl[3]
                    Q_offsets_adj = Q_base[2]

                    Q_offsets = 0.2 * (1 + torch.tanh(Q_offsets_adj.contiguous().view(-1, self.n_offsets, 3)))

                    Q_offsets = Q_offsets.view(-1)
                    mean_offsets = mean_offsets.contiguous().view(-1)
                    scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)

                    mask_slice = mask_Gaussian_curr[N_start:N_end]
                    mask_slice = mask_slice.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
                    offsets_decoded_tmp = decoder_gaussian_chunk(mean_offsets[mask_slice], scale_offsets[mask_slice], Q_offsets[mask_slice], file_name=offsets_b_name, chunk_size=3_0000)
                    offsets_slice_q = torch.zeros_like(mean_offsets)
                    offsets_slice_q[mask_slice] = offsets_decoded_tmp
                    offsets_slice_q = offsets_slice_q.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]

                    offsets_slice_q_list.append(offsets_slice_q)

                _grid_offsets_curr = torch.cat(offsets_slice_q_list, dim=0)  # [N_curr, 10, 3]

                offsets_canvas_valid[mask_anchor_curr] += _grid_offsets_curr
            dec_time_ed_ttl[ss] = get_time()

        final_anchor_mask = _mask_anchor_canvas_valid[:, decode_until-1]

        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        self._anchor_feat = nn.Parameter(feat_canvas_valid.float()[final_anchor_mask])
        self._offset = nn.Parameter(offsets_canvas_valid.float()[final_anchor_mask])

        self._anchor = nn.Parameter(_anchor_valid.float()[final_anchor_mask])
        self._scaling = nn.Parameter(scaling_canvas_valid.float()[final_anchor_mask])
        self._mask = nn.Parameter(_mask_Gaussian_canvas_valid.float()[final_anchor_mask])

        self.decoded_version = True

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D + len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(
                    hash_embeddings[len_3D + len_2D:len_3D + len_2D * 2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(
                    hash_embeddings[len_3D + len_2D * 2:len_3D + len_2D * 3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        print('Parameters are successfully replaced by decoded ones!')

        ss = decode_until - 1

        log_info = f"\n[ITER final Step {ss}] Decoded time in s: " \
                   f"head_time {round(head_time_ed - head_time_st, 6) if ss==0 else 0}, " \
                   f"dec_inc_time {round(dec_time_inc_ed_ttl[ss] - dec_time_inc_st_ttl[ss], 6) if ss>0 else 0}, " \
                   f"ttl_time (w/head w/inc) {round(dec_time_ed_ttl[ss] - dec_time_st_ttl[ss], 6)}"

        return log_info
