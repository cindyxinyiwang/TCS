import os
import sys
import time
import gc

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init



def memReport():
  for obj in gc.get_objects():
    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
      print(type(obj), obj.size())

def get_criterion(hparams):
  loss_reduce = False
  crit = nn.CrossEntropyLoss(ignore_index=hparams.pad_id, size_average=False, reduce=loss_reduce)
  if hparams.cuda:
    crit = crit.cuda()
  return crit

def get_performance(crit, logits, labels, hparams, sum_loss=True, logits_q=None, batch_size=None):
  if logits_q is not None:
    _, trg_vocab_size = logits.size()
    loss_p = crit(logits, labels).view(batch_size, -1).sum(-1)
    loss_q = crit(logits_q, labels).view(batch_size, -1).sum(-1)
    weight = torch.exp(loss_p.data - loss_q.data)
    ones = torch.FloatTensor([1]).expand_as(weight)
    if hparams.cuda: ones = ones.cuda()
    weight = torch.min(weight, ones)
    if hparams.mask_weight > 0:
      mask = weight <= hparams.mask_weight
      weight.masked_fill_(mask, 0)
    loss = loss_p.view(batch_size, -1) * weight.unsqueeze(1)
    loss = loss.view(-1) 
  else:
    loss = crit(logits, labels)
  mask = (labels == hparams.pad_id)
  _, preds = torch.max(logits, dim=1)
  acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum()
  if sum_loss: loss = loss.sum()
  return loss, acc

def count_params(params):
  num_params = sum(p.data.nelement() for p in params)
  return num_params

def save_checkpoint(extra, model, optimizer, hparams, path, model_q=None, optimizer_q=None):
  print("Saving model to '{0}'".format(path))
  torch.save(extra, os.path.join(path, "extra.pt"))
  torch.save(model, os.path.join(path, "model.pt"))
  if model_q is not None:
    torch.save(model_q, os.path.join(path, "model_q.pt"))
  torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
  if optimizer_q is not None:
    torch.save(optimizer_q.state_dict(), os.path.join(path, "optimizer_q.pt"))
  torch.save(hparams, os.path.join(path, "hparams.pt"))

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

def set_lr(optim, lr):
  for param_group in optim.param_groups:
    param_group["lr"] = lr

def init_param(p, init_type="uniform", init_range=None):
  if init_type == "xavier_normal":
    init.xavier_normal(p)
  elif init_type == "xavier_uniform":
    init.xavier_uniform(p)
  elif init_type == "kaiming_normal":
    init.kaiming_normal(p)
  elif init_type == "kaiming_uniform":
    init.kaiming_uniform(p)
  elif init_type == "uniform":
    #assert init_range is not None and init_range > 0
    init.uniform_(p, -init_range, init_range)
  else:
    raise ValueError("Unknown init_type '{0}'".format(init_type))


def get_attn_subsequent_mask(seq, pad_id=0):
  """ Get an attention mask to avoid using the subsequent info."""

  assert seq.dim() == 2
  batch_size, max_len = seq.size()
  sub_mask = torch.triu(
    torch.ones(max_len, max_len), diagonal=1).unsqueeze(0).repeat(
      batch_size, 1, 1).type(torch.ByteTensor)
  if seq.is_cuda:
    sub_mask = sub_mask.cuda()
  return sub_mask

def grad_clip(params, grad_bound=None):
  """Clipping gradients at L-2 norm grad_bound. Returns the L-2 norm."""

  params = list(filter(lambda p: p.grad is not None, params))
  total_norm = 0
  for p in params:
    if p.grad is None:
      continue
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm ** 2
  total_norm = total_norm ** 0.5

  if grad_bound is not None:
    clip_coef = grad_bound / (total_norm + 1e-6)
    if clip_coef < 1:
      for p in params:
        p.grad.data.mul_(clip_coef)
  return total_norm

def get_grad_cos(model, data, crit):
  i = 0
  step = 0 
  grads = []
  dists = [100 for _ in range(model.hparams.lan_size)]
  data_count = 0
  for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank) in data.next_train_select():
    assert file_idx[0] == i % model.hparams.lan_size
    i += 1
    target_words = (y_count - batch_size)
    logits = model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_train_char_sparse, y_train_char_sparse, file_idx=file_idx, step=step, x_rank=x_rank)
    logits = logits.view(-1, model.hparams.trg_vocab_size)
    labels = y_train[:,1:].contiguous().view(-1)
      
    cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, model.hparams)
    total_loss = cur_tr_loss.item()
    total_corrects = cur_tr_acc.item()
    cur_tr_loss.div_(batch_size)
    cur_tr_loss.backward()
    #print(file_idx[0])
    #params = list(filter(lambda p: p.grad is not None, model.parameters()))
    params_dict = model.state_dict()
    params =  list(model.parameters())
    #for k, v in params_dict.items():
    #  print(k)
    #  print(v[0])
    #  break
    #  print(v.size())
    #for v in model.parameters():
    #  print(v.size())
    grad = {}
    d = 0
    for k, v in params_dict.items():
      if params[d].grad is not None: 
        grad[k] = params[d].grad.data.clone()
        params[d].grad.data.zero_()
      d += 1
    grads.append(grad)
    if file_idx[0] == model.hparams.lan_size-1:
      data_count += 1
      for j in range(1, model.hparams.lan_size):
        dist = 0
        if data_count == 1:
          print(data.lans[j])
        for k in grads[0].keys():
          p0 = grads[0][k]
          p1 = grads[j][k]
          p0_unit = p0 / (p0.norm(2) + 1e-10)
          p1_unit = p1 / (p1.norm(2) + 1e-10)
          cosine = (p0_unit * p1_unit).sum()

          #if "enc" in k or "decoder.attention" in k:
          if "enc" in k:
            dist = dist + cosine.item()
          if data_count == 1:
            print("{} : {}".format(k, cosine))
        dists[j] += dist
      grads = []
      if data_count == 5:
        break
  dists = [d / data_count for d in dists]
  for j in range(1, model.hparams.lan_size):
    print(data.lans[j])
    print(dists[j])
  data.update_prob_list(dists)


def get_grad_cos_all(model, data, crit):
  i = 0
  step = 0 
  grads = []
  dists = [100 for _ in range(model.hparams.lan_size)]
  data_count = 0
  for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank) in data.next_train_select_all():
    #assert file_idx[0] == (i // 2) % model.hparams.lan_size
    i += 1
    target_words = (y_count - batch_size)
    logits = model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_train_char_sparse, y_train_char_sparse, file_idx=file_idx, step=step, x_rank=x_rank)
    logits = logits.view(-1, model.hparams.trg_vocab_size)
    labels = y_train[:,1:].contiguous().view(-1)
      
    cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, model.hparams)
    total_loss = cur_tr_loss.item()
    total_corrects = cur_tr_acc.item()
    cur_tr_loss.div_(batch_size)
    cur_tr_loss.backward()
    #print(file_idx[0])
    #params = list(filter(lambda p: p.grad is not None, model.parameters()))
    params_dict = model.state_dict()
    params =  list(model.parameters())
    #for k, v in params_dict.items():
    #  print(k)
    #  print(v[0])
    #  break
    #  print(v.size())
    #for v in model.parameters():
    #  print(v.size())
    grad = {}
    d = 0
    for k, v in params_dict.items():
      if params[d].grad is not None: 
        grad[k] = params[d].grad.data.clone()
        params[d].grad.data.zero_()
      d += 1
    grads.append(grad)
    if file_idx[0] != 0:
      data_count += 1
      data_idx = file_idx[0]
      dist = 0
      if data_count == data.ave_grad:
        print(data.lans[data_idx])
      for k in grads[0].keys():
        p0 = grads[0][k]
        p1 = grads[1][k]
        p0_unit = p0 / (p0.norm(2) + 1e-10)
        p1_unit = p1 / (p1.norm(2) + 1e-10)
        cosine = (p0_unit * p1_unit).sum()

        #if "enc" in k or "decoder.attention" in k:
        if "encoder.word_emb" in k:
          dist = dist + cosine.item()
        if data_count == data.ave_grad:
          print("{} : {}".format(k, cosine))
      dists[data_idx] += dist
      grads = []
      if file_idx[0] == model.hparams.lan_size - 1 and data_count == data.ave_grad:
        break
      if data_count == data.ave_grad: data_count = 0

  dists = [d / data_count for d in dists]
  for j in range(1, model.hparams.lan_size):
    print(data.lans[j])
    print(dists[j])
  data.update_prob_list(dists)
