# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv MAE, DPT and ConvNeXt code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# https://github.com/isl-org/DPT
# https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .multimae_utils import (Block, CrossAttention, Mlp,
                             build_2d_sincos_posemb, pair, trunc_normal_)
from .output_adapter_utils import (ConvNeXtBlock, Interpolate,
                                   make_fusion_block, make_scratch)


class SpatialOutputAdapter(nn.Module):
    """Cross-attention adapter for spatial outputs, like images or feature maps.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens_enc: Dimension of tokens coming from encoder. Can be set using init method.
    :param dim_tokens: Dimension of decoder tokens
    :param depth: Number of additional (full self-attention) transformer layers after initial cross attention and MLP
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    :param mlp_ratio: MLP hidden dim ratio
    :param num_heads: Number of attention heads
    :param qkv_bias: Set to True to enable bias
    :param drop_rate: Probability of dropping attention layer outputs
    :param attn_drop_rate: Probability of dropping attention matrix elements
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    :param use_task_queries: When set to True, adds task specific tokens from encoder (if available)
        to the corresponding query entries
    :param task: Task for which encoder tokens are added to the queries of the decoder (e.g. RGB if decoder is used for RGB)
    :param context_tasks: Tasks / modalities from the encoder. Used to create learned embeddings for each task.
    :param use_xattn: When set to True, attend to the tokens from the encoder through a cross-attention layer
    """

    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: Union[int, Tuple[int, int]],
                 dim_tokens_enc: Optional[int] = None,
                 dim_tokens: int = 256,
                 depth: int = 0,
                 learnable_pos_emb: int = False,
                 image_size: Union[int, Tuple[int]] = 224,
                 mlp_ratio: int = 4.0,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 use_task_queries: bool = True,
                 task: Optional[str] = None,
                 context_tasks: Optional[list] = None,
                 use_xattn: bool = True
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens_enc = dim_tokens_enc
        self.dim_tokens = dim_tokens
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.use_task_queries = use_task_queries
        self.task = task
        self.use_xattn = use_xattn

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if context_tasks is not None:
            self.task_embeddings = nn.ParameterDict(
                {task: nn.Parameter(torch.zeros(1, 1, self.dim_tokens)) for task in context_tasks})
            for embedding in self.task_embeddings.values():
                trunc_normal_(embedding, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))

        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if not self.learnable_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, h_posemb, w_posemb, self.dim_tokens))
            trunc_normal_(self.pos_emb, std=0.02)

        # One cross attention layer followed by MLP block, an optional transformer, and an output projection
        if self.use_xattn:
            self.decoder = CrossAttention(
                dim=self.dim_tokens, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, proj_drop=drop_rate)
            self.context_norm = norm_layer(self.dim_tokens)
            self.query_norm = norm_layer(self.dim_tokens)
            self.out_norm = norm_layer(self.dim_tokens)

            mlp_hidden_dim = int(self.dim_tokens * mlp_ratio)
            self.mlp = Mlp(in_features=self.dim_tokens, hidden_features=mlp_hidden_dim)

        # Optional full self-attention transformer layers
        if depth > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.decoder_transformer = nn.Sequential(*[
                Block(dim=self.dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
        else:
            self.decoder_transformer = nn.Identity()

        self.dim_patch = self.num_channels * self.P_H * self.P_W
        self.out_proj = nn.Linear(self.dim_tokens, self.dim_patch)

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.dim_tokens_enc = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_context = nn.Linear(self.dim_tokens_enc, self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb', 'mask_token', 'task_embeddings'}

    def generate_context_embeddings(self, input_info,
                                    bs: int,
                                    size: Tuple[int, int],
                                    device: Optional[torch.device] = None):
        context_embeddings = []
        for task, info in input_info["tasks"].items():
            if self.task_embeddings is not None and task in self.task_embeddings:
                task_emb = repeat(self.task_embeddings[task], '() () d -> b n d', b=bs, n=info['num_tokens'])
            else:
                task_emb = torch.zeros((bs, info['num_tokens'], self.dim_tokens), device=device)

            if info['has_2d_posemb']:
                pos_emb = F.interpolate(self.pos_emb, size=size, mode='bilinear', align_corners=False)
                pos_emb = rearrange(pos_emb, 'b d nh nw -> b (nh nw) d')
                assert info['num_tokens'] == pos_emb.shape[1]
                task_emb = task_emb + pos_emb

            context_embeddings.append(task_emb)

        context_embeddings = torch.cat(context_embeddings, dim=1)

        return context_embeddings

    def get_queries_and_context(self, context_tokens, input_info, ids_keep, ids_restore):
        B = context_tokens.shape[0]
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        if 'num_global_tokens' in input_info:
            context_tokens_without_global = context_tokens[:, :-input_info['num_global_tokens']]
        else:
            context_tokens_without_global = context_tokens

        # Add mask tokens
        mask_tokens = repeat(self.mask_token, '() () d -> b n d', b=B,
                             n=input_info['num_task_tokens'] - context_tokens_without_global.shape[1])
        context_with_mask = torch.cat([context_tokens_without_global, mask_tokens], dim=1)

        # Unshuffle context_with_mask
        context_with_mask = torch.gather(context_with_mask, dim=1,
                                         index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Generate context_emb and add them to context
        context_emb = self.generate_context_embeddings(input_info=input_info, bs=B, size=(N_H, N_W),
                                                       device=context_tokens.device)
        context_with_mask = context_with_mask + context_emb

        # Generate queries
        if self.use_task_queries and self.task in input_info['tasks']:
            start_idx = input_info['tasks'][self.task]['start_idx']
            end_idx = input_info['tasks'][self.task]['end_idx']
            queries = context_with_mask[:, start_idx:end_idx]
        else:
            queries = repeat(self.mask_token, '() () d -> b n d', b=B, n=N_H * N_W)
            queries_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bilinear', align_corners=False)
            queries_pos_emb = rearrange(queries_pos_emb, 'b d nh nw -> b (nh nw) d')
            queries = queries + queries_pos_emb
            if self.task_embeddings is not None and self.task in self.task_embeddings:
                queries_task_emb = repeat(self.task_embeddings[self.task], '() () d -> b n d', b=B, n=N_H * N_W)
                queries = queries + queries_task_emb

        # Unshuffle context and keep only initial context (yes, again)
        context_tokens_without_global = torch.gather(context_with_mask, dim=1,
                                                     index=ids_keep.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Add back global tokens
        if 'num_global_tokens' in input_info:
            context_tokens = torch.cat(
                [context_tokens_without_global, context_tokens[:, -input_info['num_global_tokens']:]], dim=1)
        else:
            context_tokens = context_tokens_without_global

        return queries, context_tokens

    def forward(self,
                encoder_tokens: torch.Tensor,
                input_info: Dict,
                ids_keep: torch.Tensor,
                ids_restore: torch.Tensor,
                ):
        """
        Forward pass taking output tokens from encoder and optionally a subset of them corresponding
        to this output adapter's task (needs an additional mask describing position of these tokens in the queries).

        :param encoder_tokens: Output of encoder
        :param input_info: Dictionary with information about the input modalities
        :param ids_keep: IDs of unmasked tokens (tokens given to the encoder)
        :param ids_restore: IDs to unshuffle tokens
        """
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Project encoder tokens to decoder tokens
        context_tokens = self.proj_context(encoder_tokens)

        # Get queries and context
        queries, context_tokens = self.get_queries_and_context(context_tokens, input_info, ids_keep, ids_restore)

        # Perform cross attention of queries to context tokens, followed by an MLP
        if self.use_xattn:
            x = self.decoder(self.query_norm(queries), self.context_norm(context_tokens))
            x = x + self.mlp(self.out_norm(x))
        else:
            x = queries

        # Optional transformer layers if depth > 0
        x = self.decoder_transformer(x)

        # Project each token to (C * P_H * P_W)
        x = self.out_proj(x)

        # Reshape sequence of patches into image
        x = rearrange(
            x, 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
            nh=N_H, nw=N_W, ph=self.P_H, pw=self.P_W, c=self.num_channels
        )

        return x


class LinearOutputAdapter(nn.Module):
    """
    Linear output adapter.

    :param num_classes: Number of classes
    :param dim_tokens_enc: Dimension of tokens from the encoder
    :param use_mean_pooling: When set to True, uses mean pooling before linear classification head.
        Otherwise, use last token (usually the global token)
    :param norm_layer: Normalization layer
    :param init_scale: Initialization scale for linear classification head
    """

    def __init__(self,
                 num_classes: int,
                 dim_tokens_enc: Optional[int] = None,
                 use_mean_pooling: bool = True,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 init_scale: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dim_tokens_enc = dim_tokens_enc
        self.use_mean_pooling = use_mean_pooling
        self.norm_layer = norm_layer
        self.init_scale = init_scale

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.dim_tokens_enc = dim_tokens_enc

        self.norm = self.norm_layer(self.dim_tokens_enc)
        self.head = nn.Linear(dim_tokens_enc, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.init(dim_tokens_enc=self.dim_tokens_enc)

    def forward(self,
                encoder_tokens: torch.Tensor,
                **kwargs):

        if self.use_mean_pooling:
            x = encoder_tokens[:, :-1, :].mean(dim=1)
        else:
            # Global token is added at the end
            x = encoder_tokens[:, -1]

        x = self.head(self.norm(x))
        return x
