# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-modal Architecture models.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts

Code based on the implementation of "Mixture of Embedding Experts":
https://github.com/antoine77340/Mixture-of-Embedding-Experts
"""

import collections
import itertools
import logging
import re
import types

from base import BaseModel
from model.bert import BertModel
from model.lstm import LSTMModel
from model.net_vlad import NetVLAD
from model.txt_embeddings import TxtEmbeddings
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertModel as TxtBertModel
from utils.util import get_len_sequences

logger = logging.getLogger(__name__)


class MMTwins(BaseModel):
  """Whole cross-modal architecture."""

  def __init__(self,
               l2renorm,
               expert_dims,
               tokenizer,
               keep_missing_modalities,
               test_caption_mode,
               freeze_weights=False,
               mimic_ce_dims=False,
               concat_experts=False,
               concat_mix_experts=False,
               use_experts='origfeat',
               txt_inp=None,
               txt_agg=None,
               txt_pro=None,
               txt_wgh=None,
               vid_inp=None,
               vid_cont=None,
               vid_wgh=None,
               pos_enc=None,
               out_tok=None,
               use_mask='nomask',
               same_dim=512,
               vid_bert_params=None,
               txt_bert_params=None,
               agg_dims=None,
               normalize_experts=True):
    super().__init__()

    self.sanity_checks = False
    modalities = list(expert_dims.keys())
    self.expert_dims = expert_dims
    self.modalities = modalities
    logger.debug(self.modalities)
    self.mimic_ce_dims = mimic_ce_dims
    self.concat_experts = concat_experts
    self.concat_mix_experts = concat_mix_experts
    self.test_caption_mode = test_caption_mode
    self.freeze_weights = freeze_weights
    self.use_experts = use_experts
    self.use_mask = use_mask
    self.keep_missing_modalities = keep_missing_modalities
    self.l2renorm = l2renorm
    self.same_dim = same_dim
    self.txt_inp = txt_inp
    self.txt_agg = txt_agg
    self.txt_pro = txt_pro
    self.txt_wgh = txt_wgh
    self.vid_inp = vid_inp
    self.vid_cont = vid_cont
    self.vid_wgh = vid_wgh
    self.pos_enc = pos_enc
    self.out_tok = out_tok
    self.vid_bert_params = vid_bert_params
    self.normalize_experts = normalize_experts

    self.rep_dim = 4096
    self.proj_dim = 4096

    self.video_dim_reduce = nn.ModuleDict()
    for mod in self.modalities:
      in_dim = expert_dims[mod]['dim']
      if self.vid_inp in ['agg', 'both', 'all', 'temp']:
        self.video_dim_reduce[mod] = ReduceDim(in_dim, same_dim)

    # Only Bert architecture is employed for video
    if self.vid_cont == 'bert':
      vid_bert_config = types.SimpleNamespace(**vid_bert_params)
      self.vid_bert = BertModel(vid_bert_config)

    # Only Bert architecture is employed for text or caption
    if self.txt_agg[:4] in ['bert']:
      z = re.match(r'bert([a-z]{3})(\d*)(\D*)', txt_agg)
      assert z
      state = z.groups()[0]
      freeze_until = z.groups()[1]

      # Post aggregation: Use [CLS] token ("cls") or aggregate all tokens
      # (mxp, mnp)
      if z.groups()[2] and z.groups()[2] != 'cls':
        self.post_agg = z.groups()[2]
      else:
        self.post_agg = 'cls'

      if state in ['ftn', 'frz']:
        # State is finetune or frozen, we use a pretrained bert model
        txt_bert_config = 'bert-base-cased'

        # Overwrite config
        if txt_bert_params is None:
          dout_prob = vid_bert_params['hidden_dropout_prob']
          txt_bert_params = {
              'hidden_dropout_prob': dout_prob,
              'attention_probs_dropout_prob': dout_prob,
          }
        self.txt_bert = TxtBertModel.from_pretrained(txt_bert_config,
                                                     **txt_bert_params)

        if state == 'frz':
          if freeze_until:
            # Freeze only certain layers
            freeze_until = int(freeze_until)
            logger.debug('Freezing text bert until layer %d excluded',
                         freeze_until)
            # Freeze net until given layer
            for name, param in self.txt_bert.named_parameters():
              module = name.split('.')[0]
              if name.split('.')[2].isdigit():
                layer_nb = int(name.split('.')[2])
              else:
                continue
              if module == 'encoder' and layer_nb in range(freeze_until):
                param.requires_grad = False
                logger.debug(name)
          else:
            # Freeze the whole model
            for name, param in self.txt_bert.named_parameters():
              module = name.split('.')[0]
              if module == 'encoder':
                param.requires_grad = False
        else:
          assert not freeze_until

      if self.txt_inp == 'bertfrz':
        # Freeze model
        for param in self.txt_bert.embeddings.parameters():
          param.requires_grad = False
      elif self.txt_inp not in ['bertftn']:
        logger.error('Wrong parameter for the text encoder')
      text_dim = self.txt_bert.config.hidden_size
    else:
      raise "Only Bert architecture can be employed for text"

    # compute representations
    self.txt_avg2rep = nn.Linear(text_dim, self.rep_dim, bias=False)
    self.vid_avg2rep = nn.Linear(same_dim, self.rep_dim, bias=False)

    # projection head for contrastive learning
    sizes = [self.rep_dim, self.proj_dim]
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    self.txt_projector = nn.Sequential(*layers)
    # self.txt_projector = nn.Sequential()
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    self.vid_projector = nn.Sequential(*layers)
    #self.vid_projector = nn.Sequential()

    # normalization layer for the representations z1 and z2
    self.bn = nn.BatchNorm1d(self.proj_dim, affine=False)

    self.debug_dataloader = False
    if self.debug_dataloader:
      self.tokenizer = tokenizer

  def compute_weights_from_emb(self, embd):
    # Compute the modality weights given an embedding

    # vid emb
    if len(embd.size()) == 2:
      embd = self.moe_vid_dropout(embd)
      moe_weights = th.cat(
          [self.moe_fc_vid[mod](embd) for mod in self.modalities], dim=-1)
      moe_weights = F.softmax(moe_weights, dim=1)

    # text emb
    elif len(embd.size()) == 3:
      embd = self.moe_txt_dropout(embd)
      b, k, d = embd.size()
      m = len(self.modalities)
      embd = embd.view(b * k, d)
      moe_weights = th.cat(
          [self.moe_fc_txt[mod](embd) for mod in self.modalities], dim=-1)
      moe_weights = F.softmax(moe_weights, dim=1)
      moe_weights = moe_weights.view(b, k, m)

    return moe_weights

  def compute_weights_from_norm(self, embds):
    # Compute the modality weights according to their norm

    device = embds[self.modalities[0]].device
    # vid emb
    if len(embds[self.modalities[0]].size()) == 2:
      b, d = embds[self.modalities[0]].size()

    # text emb
    elif len(embds[self.modalities[0]].size()) == 3:
      b, k, d = embds[self.modalities[0]].size()
      for idx, mod in self.modalities:
        embds[mod] = embds[mod].view(b * k, d)
      b = b * k

    m = len(self.modalities)
    norm_embd = th.zeros(b, m).to(device)
    for idx, mod in enumerate(self.modalities):
      norm_embd[:, idx] = th.norm(embds[mod], p=2, dim=1)

    sum_norm = th.sum(norm_embd, dim=1)  # b
    sum_norm = sum_norm.unsqueeze(1)  # b x 1

    weights = th.div(norm_embd, sum_norm)

    return weights

  def forward(self,
              token_ids,
              features,
              features_t,
              features_ind,
              features_avgpool,
              features_maxpool,
              query_masks,
              out='conf',
              device=None,
              debug=None):

    self.device = device
    experts_feats = features
    experts_feats_t = features_t
    experts_feats_ind = features_ind
    ind = {}
    for mod in self.modalities:
      ind[mod] = th.max(experts_feats_ind[mod], 1)[0]
    pooled_experts = {}

    for _, mod in enumerate(self.modalities):
      pooled_experts[f'{mod}_avgpool'] = features_avgpool[mod]
      pooled_experts[f'{mod}_maxpool'] = features_maxpool[mod]

    # Notation: B = batch size, M = number of modalities

    # Output experts
    experts = collections.OrderedDict()

    # Unroll repeated captions into present minibatch
    b, captions_per_video, max_text_words, feat_dim = token_ids.size()
    m = len(self.modalities)

    # If Bert architecture is employed
    if self.txt_agg[:4] in ['bert']:
      token_ids = token_ids.view(b * captions_per_video, max_text_words,
                                 feat_dim)

      input_ids_list = []
      token_type_ids_list = []  # Modality id
      position_ids_list = []  # Position
      attention_mask_list = []  # Valid token or not

      ids_size = (b * captions_per_video,)

      for pos_id in range(max_text_words):
        input_ids_list.append(token_ids[:, pos_id, 0].to(dtype=th.long))
        token_type_ids_list.append(th.full(ids_size, 0, dtype=th.long))
        position_ids_list.append(th.full(ids_size, pos_id, dtype=th.long))
        attention_mask_list.append(token_ids[:, pos_id, 1].to(dtype=th.long))

      input_ids = th.stack(input_ids_list, dim=1).to(device)
      token_type_ids = th.stack(token_type_ids_list, dim=1).to(device)
      position_ids = th.stack(position_ids_list, dim=1).to(device)
      attention_mask = th.stack(attention_mask_list, dim=1).to(device)

      txt_bert_output = self.txt_bert(input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      head_mask=None)
      last_layer = txt_bert_output[0]  # [b, max_len, 768]

      # aggregate the output of last layer
      text = last_layer[:, 0]  # [b, 768]
      embeddings = last_layer[:, 1:]
      text_avg = th.mean(embeddings, 1)  # [b, 768]

    if self.vid_inp in ['agg', 'both', 'all']:
      agg_experts = collections.OrderedDict()
      mnp_experts = collections.OrderedDict()
      maxp_experts = collections.OrderedDict()

      # Embed all features to a common dimension
      for mod in self.modalities:
        layer = self.video_dim_reduce[mod]
        mnp_experts[mod] = layer(pooled_experts[f'{mod}_avgpool'])
        maxp_experts[mod] = layer(pooled_experts[f'{mod}_maxpool'])

      for mod in self.modalities:
        agg_experts[mod] = maxp_experts[mod]

    if self.vid_inp in ['both', 'temp', 'all']:
      for mod in self.modalities:
        layer = self.video_dim_reduce[mod]
        experts_feats[mod] = layer(experts_feats[mod])

    # If Bert architecture is employed
    if self.vid_cont == 'bert':
      # 0=[CLS] 1=[SEP] 2=[AGG] 3=[MAXP] 4=[MNP] 5=[VLAD] 6=[FEA]
      input_ids_list = []
      token_type_ids_list = []  # Modality id
      # Position (0 = no position, 1 = unknown, >1 = valid position)
      position_ids_list = []
      features_list = []  # Semantics
      attention_mask_list = []  # Valid token or not

      modality_to_tok_map = collections.OrderedDict()

      # [CLS] token
      tok_id = 0
      ids_size = (b,)
      input_ids_list.append(th.full(ids_size, 0, dtype=th.long))
      token_type_ids_list.append(th.full(ids_size, 0, dtype=th.long))
      position_ids_list.append(th.full(ids_size, 0, dtype=th.long).to(device))
      features_list.append(
          th.full((b, self.same_dim), 0, dtype=th.float).to(device))
      attention_mask_list.append(th.full(ids_size, 1, dtype=th.long).to(device))

      # Number of temporal tokens per modality
      if self.vid_inp in ['temp', 'both', 'all']:
        max_expert_tokens = collections.OrderedDict()
        for _, modality in enumerate(self.modalities):
          max_expert_tokens[modality] = experts_feats[modality].size()[1]

      # Make the features_t and raw_captions_t start at the minimum value
      if self.pos_enc == 'tint':

        # Clamp the position encoding to [0, max_position_embedding - 1]
        max_pos = self.vid_bert_params['max_position_embeddings'] - 1
        for _, modality in enumerate(self.modalities):
          experts_feats_t[modality].clamp_(min=0, max=max_pos)
          experts_feats_t[modality] = experts_feats_t[modality].long().to(
              device)

      for _, modality in enumerate(self.modalities):
        token_type = self.expert_dims[modality]['idx']

        # Add an aggregated feature token
        if self.vid_inp in ['agg', 'both', 'all']:
          tok_id += 1
          modality_to_tok_map[modality] = tok_id
          input_ids_list.append(th.full(ids_size, 2, dtype=th.long))
          token_type_ids_list.append(
              th.full(ids_size, token_type, dtype=th.long))
          position_ids_list.append(
              th.full(ids_size, 0, dtype=th.long).to(device))
          if self.out_tok == 'sep':
            features_list.append(
                th.full((b, self.same_dim), 0, dtype=th.float).to(device))
          elif self.out_tok == 'mxp':
            features_list.append(maxp_experts[modality])
          elif self.out_tok == 'mnp':
            features_list.append(mnp_experts[modality])
          attention_mask_list.append(ind[modality].to(dtype=th.long).to(device))
        if self.vid_inp in ['temp', 'both', 'all']:
          for frame_id in range(max_expert_tokens[modality]):
            if self.pos_enc == 'ordr':
              position_ids_list.append(
                  th.full(ids_size, frame_id + 1, dtype=th.long).to(device))
            elif self.pos_enc == 'tint':
              position_ids_list.append(experts_feats_t[modality][:, frame_id])
            elif self.pos_enc == 'type':
              position_ids_list.append(
                  th.full(ids_size, 1, dtype=th.long).to(device))
            tok_id += 1
            input_ids_list.append(th.full(ids_size, 6, dtype=th.long))
            token_type_ids_list.append(
                th.full(ids_size, token_type, dtype=th.long))
            features_list.append(experts_feats[modality][:, frame_id, :])
            attention_mask_list.append(
                experts_feats_ind[modality][:, frame_id].to(dtype=th.long))

      features = th.stack(features_list, dim=1).to(self.device)
      input_ids = th.stack(input_ids_list, dim=1).to(self.device)
      token_type_ids = th.stack(token_type_ids_list, dim=1).to(self.device)
      if self.pos_enc != 'none':
        position_ids = th.stack(position_ids_list, dim=1).to(self.device)
      else:
        position_ids = None
      attention_mask = th.stack(attention_mask_list, dim=1).to(self.device)

      if self.debug_dataloader and debug:
        token_ids = token_ids.view(b, captions_per_video, max_text_words,
                                   feat_dim)
        self.display_minibatch(token_ids, input_ids, attention_mask,
                               token_type_ids, position_ids, features)
        token_ids = token_ids.view(b * captions_per_video, max_text_words,
                                   feat_dim)

      # [CLS] {[AGG1][FEA1_1]...[FEA1_30]} {[AGG2][FEA2_1]...[FEA2_30]} ... {[AGG7][FEA7_1]...[FEA7_30]}
      vid_bert_output = self.vid_bert(input_ids, 
                                      attention_mask=attention_mask,  # []
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      features=features)

      last_layer = vid_bert_output[0]
      vid_embd = last_layer[:, 0]
      embeddings = last_layer[:, 1:]
      vid_avg = th.mean(embeddings, 1)  # [b, 768]

      for _, modality in enumerate(self.modalities):
        experts[modality] = last_layer[:, modality_to_tok_map[modality]]

    if self.vid_wgh == 'nrm':
      vid_weights = self.compute_weights_from_norm(experts)
    elif self.vid_wgh == 'emb':
      vid_weights = self.compute_weights_from_emb(vid_embd)
    elif self.vid_wgh == 'none':
      vid_weights = th.ones(b, m).to(device)
    else:
      msg = 'video weighting mode {} not supported'
      raise NotImplementedError(msg.format(self.vid_wgh))

    if not self.keep_missing_modalities:
      # Zero padding of the missing modalities
      available = th.zeros(b, m).to(device)
      for idx, mod in enumerate(self.modalities):
        available[:, idx] = ind[mod].float()  # B x M
      vid_weights = vid_weights * available

    # Normalize video weights so that they sum to one
    vid_weights = nn.functional.normalize(vid_weights, p=1, dim=-1)

    if self.txt_wgh == 'emb':
      text_weights = self.compute_weights_from_emb(text)
    elif self.txt_wgh == 'none':
      text_weights = th.ones(b, captions_per_video, m).to(device)
    else:
      msg = 'txt weighting mode {} not supported'
      raise NotImplementedError(msg.format(self.txt_wgh))

    # Normalize text weights so that they sum to one
    text_weights = nn.functional.normalize(text_weights, p=1, dim=-1)

    # compute video and text represnetations
    vid_reps = self.vid_avg2rep(vid_avg)
    txt_reps = self.txt_avg2rep(text_avg)

    # projection for contrastive learning
    vid_embd_CL = self.bn(self.vid_projector(vid_reps))
    txt_embd_CL = self.bn(self.vid_projector(txt_reps))

    if out == 'conf':  # Output confusion matrix
      cross_view_conf_matrix = sharded_cross_view_inner_product(
          vid_rep=vid_reps,
          txt_rep=txt_reps,
          subspaces=self.modalities
      )
      return {
          'modalities': self.modalities,
          'cross_view_conf_matrix': cross_view_conf_matrix,
      }
    elif out == 'corr':
      corrletaion_matrix = cpt_cross_corrletaion_matrix(
          vid_embd_CL=vid_embd_CL,
          txt_embd_CL=txt_embd_CL,
          subspaces=self.modalities
      )
      return {
          'modalities': self.modalities,
          'corrletaion_matrix': corrletaion_matrix,
      }
    else:  # Output the embeddings
      return {
          'vid_reps': vid_reps,
          'text_reps': txt_reps,
      }

  def display_minibatch(self, token_ids, input_ids, attention_mask,
                        token_type_ids, position_ids, features):
    for i in range(1):
      logger.debug()
      # logger.debug(f'Sample {i}:')
      logger.debug('Text:')
      ids = token_ids[i, 0, :, 0].cpu().numpy()
      logger.debug(ids)

      tokens = self.tokenizer.convert_ids_to_tokens(ids)
      logger.debug(tokens)

      logger.debug('Video:')
      # logger.debug(f'input_ids: {input_ids[i]}')
      # logger.debug(f'attention_mask: {attention_mask[i]}')
      # logger.debug(f'token_type_ids: {token_type_ids[i]}')
      # logger.debug(f'position_ids: {position_ids[i]}')
      logger.debug(features[i].shape)


class GatedEmbeddingUnit(nn.Module):
  """Gated embedding module.

  as described in
  "Learning a Text-Video Embedding from Incomplete and Heterogeneous Data"
  """

  def __init__(self, input_dimension, output_dimension, use_bn, normalize):
    super(GatedEmbeddingUnit, self).__init__()

    self.fc = nn.Linear(input_dimension, output_dimension)
    self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)
    self.normalize = normalize

  def forward(self, x):
    x = self.fc(x)
    x = self.cg(x)
    if self.normalize:
      x = F.normalize(x, dim=-1)
    return x


class MimicCEGatedEmbeddingUnit(nn.Module):

  def __init__(self, input_dimension, output_dimension, use_bn):
    super().__init__()
    self.cg = ContextGating(input_dimension, add_batch_norm=use_bn)

  def forward(self, x):
    x = self.cg(x)
    x = F.normalize(x, dim=-1)
    return x


class ReduceDim(nn.Module):

  def __init__(self, input_dimension, output_dimension):
    super(ReduceDim, self).__init__()
    self.fc = nn.Linear(input_dimension, output_dimension)

  def forward(self, x):
    x = self.fc(x)
    x = F.normalize(x, dim=-1)
    return x


class GatedLinearUnit(nn.Module):

  def forward(self, x, mask):
    x = th.cat((x, mask), 1)
    return F.glu(x, 1)


class ContextGating(nn.Module):
  """Context gating class."""

  def __init__(self, dimension, add_batch_norm=True):
    super(ContextGating, self).__init__()
    self.fc = nn.Linear(dimension, dimension)
    self.add_batch_norm = add_batch_norm
    self.batch_norm = nn.BatchNorm1d(dimension)

  def forward(self, x):
    x1 = self.fc(x)
    if self.add_batch_norm:
      x1 = self.batch_norm(x1)
    x = th.cat((x, x1), 1)  # x [b, 1024]
    return F.glu(x, 1)  # return [b, 512]


class GatedEmbeddingUnitReasoning(nn.Module):

  def __init__(self, output_dimension):
    super(GatedEmbeddingUnitReasoning, self).__init__()
    self.cg = ContextGatingReasoning(output_dimension)

  def forward(self, x, mask):
    x = self.cg(x, mask)
    x = F.normalize(x, dim=-1)
    return x


class ContextGatingReasoning(nn.Module):
  """Context gating reasoning class."""

  def __init__(self, dimension, add_batch_norm=True):
    super(ContextGatingReasoning, self).__init__()
    self.fc = nn.Linear(dimension, dimension)
    self.add_batch_norm = add_batch_norm
    self.batch_norm = nn.BatchNorm1d(dimension)
    self.batch_norm2 = nn.BatchNorm1d(dimension)

  def forward(self, x, x1):

    x2 = self.fc(x)

    if self.add_batch_norm:
      x1 = self.batch_norm(x1)
      x2 = self.batch_norm2(x2)

    t = x1 + x2

    x = th.cat((x, t), 1)
    return F.glu(x, 1)


def sharded_cross_view_inner_product(vid_rep,
                                     txt_rep,
                                     subspaces):
  """Compute similarities between all captions and videos."""
  
  txt_rep_norm = F.normalize(txt_rep, dim=1)
  vid_rep_norm = F.normalize(vid_rep, dim=1)
  sims = txt_rep_norm @ vid_rep_norm.T
  return sims


def cpt_cross_corrletaion_matrix(vid_embd_CL,
                                 txt_embd_CL,
                                 subspaces):
  """Compute cross-correlation matrix."""

  #vid_embd_CL_norm = F.normalize(vid_embd_CL, dim=1)
  #txt_embd_CL_norm = F.normalize(txt_embd_CL, dim=1)

  cross_corr_mat = vid_embd_CL.T @ txt_embd_CL
  return cross_corr_mat