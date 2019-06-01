#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import sys

import numpy as np

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import init_param


class PositionalEmbedding(nn.Module):
  def __init__(self, hparams):
    super(PositionalEmbedding, self).__init__()

    self.hparams = hparams

    if self.hparams.pos_emb_size is not None:
      self.emb = nn.Embedding(self.hparams.pos_emb_size,
                              self.hparams.d_word_vec,
                              padding_idx=0)
      if self.hparams.cuda:
        self.emb = self.emb.cuda()
    else:
      d_word_vec = self.hparams.d_word_vec
      self.emb_scale = self.hparams.init_range * d_word_vec
      freq = torch.arange(0, d_word_vec, 2).float() / d_word_vec
      self.freq = 1.0 / (10000.0 ** Variable(freq))
      #self.freq = 10000.0 ** Variable(freq)
      #print(self.freq)
      if self.hparams.cuda:
        self.freq = self.freq.cuda()

  def forward(self, data=None, pos=None):
    """Compute positional embeddings.

    Args:
      x: Tensor of size [batch_size, max_len]

    Returns:
      emb: Tensor of size [batch_size, max_len, d_word_vec].
    """

    d_word_vec = self.hparams.d_word_vec
    if pos is not None:
      batch_size, max_len = pos.size()
      pos = Variable(pos)
    else:
      batch_size, max_len = data
      pos = Variable(torch.arange(0, max_len))
    if self.hparams.cuda:
      pos = pos.cuda()
    if self.hparams.pos_emb_size is not None:
      pos = pos.add_(1).long().unsqueeze(0).expand_as(x).contiguous()
      emb = self.emb(pos)
    else:
      emb = pos.float().unsqueeze(-1) * self.freq.unsqueeze(0)
      sin = torch.sin(emb).mul_(self.emb_scale).unsqueeze(-1)
      cos = torch.cos(emb).mul_(self.emb_scale).unsqueeze(-1)
      #emb = pos.float().unsqueeze(-1) / self.freq.unsqueeze(0)
      #sin = torch.sin(emb).unsqueeze(-1)
      #cos = torch.cos(emb).unsqueeze(-1)
      emb = torch.cat([sin, cos], dim=-1).contiguous().view(max_len, d_word_vec)
      emb = emb.unsqueeze(0).expand(batch_size, -1, -1)

    return emb


class LayerNormalization(nn.Module):
  def __init__(self, d_hid, hparams, eps=1e-9):
    super(LayerNormalization, self).__init__()

    self.d_hid = d_hid
    if hasattr(hparams, "layernorm_eps"):
      self.eps = hparams.layernorm_eps
    else:
      self.eps = eps
    self.scale = nn.Parameter(torch.ones(self.d_hid), requires_grad=True)
    self.offset= nn.Parameter(torch.zeros(self.d_hid), requires_grad=True)

  def forward(self, x):
    assert x.dim() >= 2
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return self.scale * (x - mean) / (std + self.eps) + self.offset


class ScaledDotProdAttn(nn.Module):
  def __init__(self, hparams):
    super(ScaledDotProdAttn, self).__init__()
    self.temp = np.power(hparams.d_model, 0.5)
    self.dropout = nn.Dropout(hparams.dropout)
    self.softmax = nn.Softmax(dim=2)
    self.hparams = hparams

  def forward(self, q, k, v, attn_mask=None):
    """Compute Softmax(q * k.T / sqrt(dim)) * v

    Args:
      q: [batch_size, len_q, d_q].
      k: [batch_size, len_k, d_k].
      v: [batch_size, len_v, d_v].
    
    Note: batch_size may be n_heads * batch_size, but we don't care.
    
    Must have:
      d_q == d_k
      len_k == len_v

    Returns:
      attn: [batch_size, len_q, d_v].
    """
    if q.dim() == 4:
      # batch_size, n_heads, len, dim
      batch_q, len_q, d_q, n_heads = q.size()
      batch_k, len_k, d_k, n_heads = k.size()
      batch_v, len_v, d_v, n_heads = v.size()
    
      # batch_size, len_q, len_k, n_heads
      attn = torch.einsum("bidn,bjdn->bijn", (q, k)) / self.temp
      # attn_mask: [batch_size, len_q, len_k]
      if attn_mask is not None:
        attn.data.masked_fill_(attn_mask.unsqueeze(3), -self.hparams.inf)
      attn = self.softmax(attn).contiguous()
      attn = self.dropout(attn)
      output = torch.einsum("bijn,bjdn->bidn", (attn, v)).contiguous().view(batch_q, len_q, -1)
      return output

    batch_q, len_q, d_q = q.size()
    batch_k, len_k, d_k = k.size()
    batch_v, len_v, d_v = v.size()

    assert batch_q == batch_k and batch_q == batch_v
    assert d_q == d_k and len_k == len_v

    # [batch_size, len_q, len_k]
    attn = torch.bmm(q, k.transpose(1, 2)) / self.temp

    # attn_mask: [batch_size, len_q, len_k]
    if attn_mask is not None:
      #attn.data.masked_fill_(attn_mask, -float("inf"))
      attn.data.masked_fill_(attn_mask, -self.hparams.inf)
    size = attn.size()
    assert len(size) > 2 and len_q == size[1] and len_k == size[2]

    # softmax along the len_k dimension
    # [batch_size, len_q, len_k]
    attn = self.softmax(attn).contiguous()

    # [batch_size, len_q, len_k == len_v]
    attn = self.dropout(attn)

    # [batch_size, len_q, d_v]
    output = torch.bmm(attn, v).contiguous()

    return output


class RelativeMultiHeadAttn(nn.Module):
  def __init__(self, hparams, enc=False, n_layer=-1):
    super(RelativeMultiHeadAttn, self).__init__()

    self.hparams = hparams
    self.set_sep = (n_layer in self.hparams.sep_layer) and enc
    self.enc = enc
    self.n_layer = n_layer
    #self.layer_norm = LayerNormalization(hparams.d_model, hparams)
    self.layer_norm = torch.nn.LayerNorm(hparams.d_model)
    self.temp = np.power(hparams.d_model, 0.5)
    self.softmax = nn.Softmax(dim=2)
    self.pos_emb = PositionalEmbedding(hparams)
    self.dropout = nn.Dropout(hparams.dropout)
    # projection of concatenated attn
    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v

    self.q = nn.Linear(d_model, n_heads * d_q, bias=False)
    self.k = nn.Linear(d_model, n_heads * d_k, bias=False)
    self.v = nn.Linear(d_model, n_heads * d_v, bias=False)
    init_param(self.q.weight, init_type="uniform", init_range=hparams.init_range)
    init_param(self.k.weight, init_type="uniform", init_range=hparams.init_range)
    init_param(self.v.weight, init_type="uniform", init_range=hparams.init_range)

    if self.hparams.sep_head_weight and self.enc:
      self.head_w = []
      for i in range(self.hparams.lan_size):
        h_w = nn.Linear(d_model, n_heads, bias=False)
        init_param(h_w.weight, init_type="uniform", init_range=hparams.init_range)
        self.head_w.append(h_w)
      self.head_w = nn.ModuleList(self.head_w)
      if self.hparams.cuda: self.head_w = self.head_w.cuda()
    
    if self.enc and self.n_layer < self.hparams.max_loc_layer:
      self.r = []

      r = nn.Linear(d_model, n_heads * d_v, bias=False)
      init_param(r.weight, init_type="uniform", init_range=hparams.init_range)
      self.r.append(r)
      self.r = nn.ModuleList(self.r)

    if self.hparams.cuda:
      self.q = self.q.cuda()
      self.k = self.k.cuda()
      self.v = self.v.cuda()
      if self.enc and self.n_layer < self.hparams.max_loc_layer:
        self.r = self.r.cuda()
    if self.hparams.relative_pos_c:
      ub = nn.Linear(d_q, 1, bias=False)
      init_param(ub.weight, init_type="uniform", init_range=hparams.init_range)
      self.ub = ub
    if self.hparams.relative_pos_d and (self.enc and self.n_layer < self.hparams.max_loc_layer):
      self.vb = []
      vb = nn.Linear(d_q, 1, bias=False)
      init_param(vb.weight, init_type="uniform", init_range=hparams.init_range)
      self.vb.append(vb)
      self.vb = nn.ModuleList(self.vb)
      if self.hparams.cuda: self.vb = self.vb.cuda()

    self.w_proj = nn.Linear(n_heads * d_v, d_model, bias=False)
    init_param(self.w_proj.weight, init_type="uniform", init_range=hparams.init_range)
    if self.hparams.cuda:
      self.w_proj = self.w_proj.cuda()
      if self.hparams.relative_pos_c:
        self.ub = self.ub.cuda()

  def forward(self, q, k, v, attn_mask=None, file_idx=None, step=None):
    """Performs the following computations:

         head[i] = Attention(q * w_q[i], k * w_k[i], v * w_v[i])
         outputs = concat(all head[i]) * self.w_proj

    Args:
      q: [batch_size, len_q, d_q].
      k: [batch_size, len_k, d_k].
      v: [batch_size, len_v, d_v].

    Must have: len_k == len_v
    Note: This batch_size is in general NOT the training batch_size, as
      both sentences and time steps are batched together for efficiency.

    Returns:
      outputs: [batch_size, len_q, d_model].
    """

    residual = q 

    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v
    batch_size, len_q, d_q = q.size()
    batch_size, len_k, d_k = k.size()
    batch_size, len_v, d_v = v.size()

    #if  self.enc and self.n_layer < self.hparams.max_loc_layer:
    if self.enc and self.n_layer < self.hparams.max_loc_layer:
      r = torch.arange(len_q-1, -len_k, -1.0).unsqueeze(0)
      # [1, len_q + len_k, d_word_vec]
      r = self.pos_emb(pos=r)
      pos_mask = torch.zeros(len_q, len_q+len_k-1)
      for i in range(len_q):
          # [batch_size, 1, len_k]
          pos_mask[i, len_q-i-1:len_q+len_k-i-1] = 1
      if self.hparams.cuda:
        pos_mask = pos_mask.cuda()
      if (not self.hparams.decode) and self.hparams.sep_relative_loc and self.set_sep and self.hparams.sep_step and step == self.hparams.sep_step:
        sep = True
        print("separating position enc params...")
        for i in range(1, self.hparams.lan_size):
          new_r = nn.Linear(d_model, n_heads * d_v, bias=False)
          new_r.weight.data = self.r[0].weight.data
          if self.hparams.cuda: new_r = new_r.cuda()
          self.r.append(new_r)
          if self.hparams.cuda: self.r = self.r.cuda()
          if self.hparams.relative_pos_d:
            new_vb = nn.Linear(d_q, 1, bias=False)
            new_vb.weight.data = self.vb[0].weight.data
            if self.hparams.cuda: new_vb = new_vb.cuda()
            self.vb.append(new_vb)
      elif self.hparams.sep_relative_loc and self.set_sep and step > self.hparams.sep_step:
        sep = True
      else:
        sep = False
      # [batch_size, len_q, len_q+len_k, n_heads]
      pos_mask = pos_mask.byte().unsqueeze(0).unsqueeze(3).expand(batch_size, -1, -1, n_heads)
      if sep:
        head_r =  self.r[file_idx[0]](r)
      else:
        head_r =  self.r[0](r)
      # batch_size, lenq+lenk, d_q, n_head
      head_r = head_r.view(1, r.size(1), -1, n_heads).expand(batch_size, -1, -1, -1)
    # batch_size, len, d_q * n_head
    head_q, head_k, head_v = self.q(q), self.k(k), self.v(v)
    # batch_size, len, dim, n_head
    head_q = head_q.view(batch_size, q.size(1), -1, n_heads)
    head_k = head_k.view(batch_size, k.size(1), -1, n_heads)
    head_v = head_v.view(batch_size, v.size(1), -1, n_heads)

    # batch_size, len_q, len_k, n_heads
    attn_a = torch.einsum("bidn,bjdn->bijn", (head_q, head_k))
    if self.enc and self.n_layer < self.hparams.max_loc_layer:
      # [batch_size, len_q, len_q + len_k, n_heads]
      attn_pos_b = torch.einsum("bidn,bjdn->bijn", (head_q, head_r))
      attn_pos_b = attn_pos_b.masked_select(pos_mask).view(batch_size, len_q, len_k, n_heads)
      attn = (attn_a + attn_pos_b)
    else:
      attn = attn_a
    if self.hparams.relative_pos_c:
      attn_c = self.ub(head_k.transpose(2, 3)).permute(0, 3, 1, 2)
      attn = attn + attn_c
    if  self.hparams.relative_pos_d and (self.enc and self.n_layer < self.hparams.max_loc_layer):
      # [batch_size, 1, len_k+len_q, n_heads]
      if sep:
        attn_pos_d = self.vb[file_idx[0]](head_r.transpose(2, 3)).permute(0, 3, 1, 2).expand(-1, len_q, -1, -1)
      else:
        attn_pos_d = self.vb[0](head_r.transpose(2, 3)).permute(0, 3, 1, 2).expand(-1, len_q, -1, -1)
      attn_pos_d = attn_pos_d.masked_select(pos_mask).view(batch_size, len_q, len_k, n_heads)
      attn = attn + attn_pos_d
    attn = attn / self.temp
    # attn_mask: [batch_size, len_q, len_k]
    if attn_mask is not None:
      attn.data.masked_fill_(attn_mask.unsqueeze(3), -self.hparams.inf)
    attn = self.softmax(attn).contiguous()
    attn = self.dropout(attn)
    outputs = torch.einsum("bijn,bjdn->bidn", (attn, head_v))
    if self.hparams.sep_head_weight and self.enc:
      # batch_size, len_q, n_head
      head_weight = torch.sigmoid(self.head_w[file_idx[0]](q))
      head_weight = head_weight.unsqueeze(2)
      outputs = outputs * head_weight
    outputs = outputs.contiguous().view(batch_size, len_q, -1)

    outputs = self.w_proj(outputs)
    outputs = self.layer_norm(outputs + residual)

    return outputs


class MultiHeadAttn(nn.Module):
  def __init__(self, hparams):
    super(MultiHeadAttn, self).__init__()

    self.hparams = hparams

    self.attention = ScaledDotProdAttn(hparams)
    self.layer_norm = LayerNormalization(hparams.d_model, hparams)

    # projection of concatenated attn
    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v
    # d_q == d_k == k_v
    self.q = nn.Linear(d_model, n_heads * d_q, bias=False)
    self.k = nn.Linear(d_model, n_heads * d_k, bias=False)
    self.v = nn.Linear(d_model, n_heads * d_v, bias=False)
    init_param(self.q.weight, init_type="uniform", init_range=hparams.init_range)
    init_param(self.k.weight, init_type="uniform", init_range=hparams.init_range)
    init_param(self.v.weight, init_type="uniform", init_range=hparams.init_range)

    # Q, K, V = [], [], []
    # for head_id in range(n_heads):
    #   q = nn.Linear(d_model, d_q, bias=False)
    #   k = nn.Linear(d_model, d_k, bias=False)
    #   v = nn.Linear(d_model, d_v, bias=False)
    #   init_param(q.weight, init_type="uniform", init_range=hparams.init_range)
    #   init_param(k.weight, init_type="uniform", init_range=hparams.init_range)
    #   init_param(v.weight, init_type="uniform", init_range=hparams.init_range)
    #   Q.append(q)
    #   K.append(k)
    #   V.append(v)
    # self.Q = nn.ModuleList(Q)
    # self.K = nn.ModuleList(K)
    # self.V = nn.ModuleList(V)
    if self.hparams.cuda:
      #self.Q = self.Q.cuda()
      #self.K = self.K.cuda()
      #self.V = self.V.cuda()
      self.q = self.q.cuda()
      self.k = self.k.cuda()
      self.v = self.v.cuda()

    self.w_proj = nn.Linear(n_heads * d_v, d_model, bias=False)
    init_param(self.w_proj.weight, init_type="uniform", init_range=hparams.init_range)
    if self.hparams.cuda:
      self.w_proj = self.w_proj.cuda()

  def forward(self, q, k, v, attn_mask=None, file_idx=None):
    """Performs the following computations:

         head[i] = Attention(q * w_q[i], k * w_k[i], v * w_v[i])
         outputs = concat(all head[i]) * self.w_proj

    Args:
      q: [batch_size, len_q, d_q].
      k: [batch_size, len_k, d_k].
      v: [batch_size, len_v, d_v].

    Must have: len_k == len_v
    Note: This batch_size is in general NOT the training batch_size, as
      both sentences and time steps are batched together for efficiency.

    Returns:
      outputs: [batch_size, len_q, d_model].
    """

    residual = q 

    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v
    batch_size = q.size(0)
    # batch_size, len, d_q * n_head
    head_q, head_k, head_v = self.q(q), self.k(k), self.v(v)
    # batch_size, len, dim, n_head
    head_q = head_q.view(batch_size, q.size(1), -1, n_heads)
    head_k = head_k.view(batch_size, k.size(1), -1, n_heads)
    head_v = head_v.view(batch_size, v.size(1), -1, n_heads)
    outputs = self.attention(head_q, head_k, head_v, attn_mask=attn_mask)
    #heads = []
    #for Q, K, V in zip(self.Q, self.K, self.V):
    #  head_q, head_k, head_v = Q(q), K(k), V(v)
    #  head = self.attention(head_q, head_k, head_v, attn_mask=attn_mask)
    #  heads.append(head)

    #outputs = torch.cat(heads, dim=-1).contiguous().view(batch_size, -1, n_heads * d_v)
    outputs = self.w_proj(outputs)
    outputs = self.layer_norm(outputs + residual)

    return outputs


class PositionwiseFF(nn.Module):
  def __init__(self, hparams):
    super(PositionwiseFF, self).__init__()
    self.hparams = hparams

    self.w_1 = nn.Linear(hparams.d_model, hparams.d_inner, bias=False)
    self.w_2 = nn.Linear(hparams.d_inner, hparams.d_model, bias=False)
    self.dropout = nn.Dropout(hparams.dropout)
    self.relu = nn.ReLU()
    #self.layer_norm = LayerNormalization(hparams.d_model, hparams)
    self.layer_norm = torch.nn.LayerNorm(hparams.d_model)

    init_param(self.w_1.weight, init_type="uniform", init_range=hparams.init_range)
    init_param(self.w_2.weight, init_type="uniform", init_range=hparams.init_range)


  def forward(self, x):
    residual = x
    batch_size, x_len, d_model = x.size()
    x = self.relu(self.w_1(x.view(-1, d_model)))
    x = self.w_2(x).view(batch_size, x_len, d_model)
    x = self.dropout(x)
    x += residual
    x = self.layer_norm(x)
    return x


class EncoderLayer(nn.Module):
  """Compose multi-head attention and positionwise feeding."""

  def __init__(self, hparams, cur_layer):
    super(EncoderLayer, self).__init__()

    self.hparams = hparams
    if self.hparams.transformer_relative_pos:
      self.attn = RelativeMultiHeadAttn(hparams, enc=True, n_layer=cur_layer)
      #self.attn = RelativeMultiHeadAttn(hparams, set_sep=(cur_layer < self.hparams.sep_layer), enc=True, n_layer=cur_layer)
      #self.attn = RelativeMultiHeadAttn(hparams, set_sep=(cur_layer >= self.hparams.n_layers - self.hparams.sep_layer))
    else:
      self.attn = MultiHeadAttn(hparams)
    self.pos_ff = PositionwiseFF(hparams)

  def forward(self, enc_input, attn_mask=None, file_idx=None, step=None):
    """Normal forward pass.

    Args:
      enc_input: [batch_size, x_len, d_model].
      attn_mask: [batch_size, x_len, x_len].
    """

    enc_output = self.attn(enc_input, enc_input, enc_input, attn_mask=attn_mask, file_idx=file_idx, step=step)
    enc_output = self.pos_ff(enc_output)
    return enc_output


class DecoderLayer(nn.Module):
  """Multi-head attention to both input_states and output_states."""

  def __init__(self, hparams, cur_layer):
    super(DecoderLayer, self).__init__()
    self.hparams = hparams
    if not self.hparams.transformer_relative_pos:
      self.y_attn = MultiHeadAttn(hparams)
      self.x_attn = MultiHeadAttn(hparams)
    else:
      self.y_attn = RelativeMultiHeadAttn(hparams, enc=True, n_layer=cur_layer)
      #self.y_attn = RelativeMultiHeadAttn(hparams, enc=False, n_layer=cur_layer)
      #self.y_attn = RelativeMultiHeadAttn(hparams, set_sep=(cur_layer >= self.hparams.n_layers - self.hparams.sep_layer))
      self.x_attn = RelativeMultiHeadAttn(hparams, enc=True, n_layer=cur_layer)
      #self.x_attn = RelativeMultiHeadAttn(hparams, set_sep=(cur_layer >= self.hparams.n_layers - self.hparams.sep_layer))
      #self.x_attn = RelativeMultiHeadAttn(hparams, set_sep=False)
    self.pos_ffn = PositionwiseFF(hparams)

  def forward(self, dec_input, enc_output, y_attn_mask=None, x_attn_mask=None, n_corrupts=0, file_idx=None, step=None):
    """Decoder.

    Args:
      y_attn_mask: self attention mask.
      x_attn_mask: decoder-encoder attention mask.
    """

    output = self.y_attn(dec_input, dec_input, dec_input, attn_mask=y_attn_mask, file_idx=file_idx, step=step)
    batch_size = dec_input.size(0)
    if n_corrupts > 0:
      #print(output)
      output = output.repeat(1, 1, n_corrupts).view(batch_size*n_corrupts, -1, self.hparams.d_model)
      #print(output)
    output = self.x_attn(output, enc_output, enc_output, attn_mask=x_attn_mask, file_idx=file_idx, step=step)
    output = self.pos_ffn(output)
    if n_corrupts > 0:
      output = output.view(-1, n_corrupts, self.hparams.d_model)
      #print(output)
      output = torch.sum(output, dim=1).div_(n_corrupts).view(batch_size, -1, self.hparams.d_model)
      #print(output)
    return output

