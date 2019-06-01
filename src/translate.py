from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import _pickle as pickle
import shutil
import gc
import os
import sys
import time
import subprocess
import numpy as np

from data_utils import DataUtil
from mult_data_utils import MultDataUtil
from hparams import *
from utils import *
from model import *
from transformer import *

import torch
import torch.nn as nn
from torch.autograd import Variable

class TranslationHparams(HParams):
  dataset = "Translate dataset"

parser = argparse.ArgumentParser(description="Neural MT translator")

parser.add_argument("--cuda", action="store_true", help="GPU or not")
parser.add_argument("--data_path", type=str, default=None, help="path to all data")
parser.add_argument("--model_dir", type=str, default="outputs", help="root directory of saved model")
parser.add_argument("--test_src_file", type=str, default=None, help="name of source test file")
parser.add_argument("--test_src_file_list", type=str, default=None, help="name of source test file")
parser.add_argument("--test_trg_file", type=str, default=None, help="name of target test file")
parser.add_argument("--test_trg_file_list", type=str, default=None, help="name of target test file")
parser.add_argument("--test_file_idx_list", type=str, default=None, help="name of target test file")
parser.add_argument("--beam_size", type=int, default=None, help="beam size")
parser.add_argument("--max_len", type=int, default=300, help="maximum len considered on the target side")
parser.add_argument("--poly_norm_m", type=float, default=0, help="m in polynormial normalization")
parser.add_argument("--non_batch_translate", action="store_true", help="use non-batched translation")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--merge_bpe", action="store_true", help="")
parser.add_argument("--src_vocab_list", type=str, default=None, help="name of source vocab file")
parser.add_argument("--trg_vocab_list", type=str, default=None, help="name of target vocab file")
parser.add_argument("--n_train_sents", type=int, default=None, help="max number of training sentences to load")
parser.add_argument("--out_file_list", type=str, default="trans", help="output file for hypothesis")
parser.add_argument("--debug", action="store_true", help="output file for hypothesis")

parser.add_argument("--nbest", action="store_true", help="whether to return the nbest list")
args = parser.parse_args()

model_file_name = os.path.join(args.model_dir, "model.pt")
if not args.cuda:
  model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
else:
  model = torch.load(model_file_name)
model.eval()

hparams_file_name = os.path.join(args.model_dir, "hparams.pt")
train_hparams = torch.load(hparams_file_name)
hparams = TranslationHparams()
for k, v in train_hparams.__dict__.items():
  setattr(hparams, k, v)

out_file_list = [os.path.join(args.model_dir, i) for i in args.out_file_list.split(",")]
print("writing translation to " + str(out_file_list))

#hparams.data_path=args.data_path
#hparams.src_vocab_list=args.src_vocab_list
#hparams.trg_vocab_list=args.trg_vocab_list
hparams.test_src_file = args.test_src_file
hparams.test_trg_file = args.test_trg_file

hparams.test_src_file_list = args.test_src_file_list.split(",")
hparams.test_trg_file_list = args.test_trg_file_list.split(",")
hparams.test_file_idx_list = [int(i) for i in args.test_file_idx_list.split(",")]
hparams.cuda=args.cuda
hparams.beam_size=args.beam_size
hparams.max_len=args.max_len
hparams.batch_size=args.batch_size
hparams.merge_bpe=args.merge_bpe
hparams.out_file_list=out_file_list
hparams.nbest=args.nbest
hparams.decode=True
if not hasattr(train_hparams, "semb_num"):
  model.hparams.semb_num = 1
if not hasattr(train_hparams, "compute_ngram"):
  model.hparams.compute_ngram = False

if hasattr(train_hparams, 'char_temp'):
  model.hparams.char_temp = train_hparams.char_temp
else:
  model.hparams.char_temp = None
if hasattr(train_hparams, 'src_char_only'):
  hparams.src_char_only = train_hparams.src_char_only
else:
  hparams.src_char_only = False
if not hasattr(train_hparams, 'n') and (train_hparams.char_ngram_n or train_hparams.char_input):
  hparams.n = 4 
if not hasattr(train_hparams, 'single_n') and (train_hparams.char_ngram_n or train_hparams.char_input):
  hparams.single_n = False 
if not hasattr(train_hparams, 'bpe_ngram'):
  hparams.bpe_ngram = False 
if not hasattr(train_hparams, 'uni'):
  hparams.uni = False 
if not hasattr(train_hparams, 'copy_mono'):
  hparams.copy_mono = False 

if not hasattr(train_hparams, "semb_num"):
  model.hparams.semb_num = 1
if not hasattr(train_hparams, "compute_ngram"):
  model.hparams.compute_ngram = False
if not hasattr(train_hparams, 'copy_mono'):
  hparams.copy_mono = False
if not hasattr(train_hparams, 'el'):
  hparams.el = False
if hasattr(train_hparams, 'detok') and train_hparams.detok:
  hparams.detok = train_hparams.detok
else:
  hparams.detok = False

model.hparams.cuda = hparams.cuda
data = MultDataUtil(hparams=hparams)
filts = [model.hparams.pad_id, model.hparams.eos_id, model.hparams.bos_id]

if not hasattr(model, 'data'):
  model.data = data
if not hasattr(hparams, 'model_type'):
  if type(model) == Seq2Seq:
    hparams.model_type = "seq2seq"
  elif type(model) == Transformer:
    hparams.model_type = 'transformer'

if args.debug:
  hparams.add_param("target_word_vocab_size", data.target_word_vocab_size)
  hparams.add_param("target_rule_vocab_size", data.target_rule_vocab_size)
  crit = get_criterion(hparams)


end_of_epoch = False
num_sentences = 0

with torch.no_grad():
  out_file = open(hparams.out_file_list[0], 'w', encoding='utf-8')

  test_idx = 0
  for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, test_file_idx, x_rank in data.next_test(test_batch_size=1):
    gc.collect()
    if hparams.model_type == 'seq2seq':
      hs = model.translate(
              x, x_mask, beam_size=args.beam_size, max_len=args.max_len, poly_norm_m=args.poly_norm_m, x_train_char=x_char, y_train_char=y_char, file_idx=test_file_idx)
    elif hparams.model_type == 'transformer': 
      hs = model.translate(
              x, x_mask, x_pos_emb_idxs, x_char_sparse_batch=x_char, beam_size=args.beam_size, max_len=args.max_len, poly_norm_m=args.poly_norm_m, file_idx=test_file_idx)
    for h in hs:
      h_best_words = map(lambda wi: data.trg_i2w[wi],
                       filter(lambda wi: wi not in filts, h))
      if hparams.merge_bpe:
        line = ''.join(h_best_words)
        line = line.replace('▁', ' ')
      else:
        line = ' '.join(h_best_words)
      line = line.strip()
      out_file.write(line + '\n')
      out_file.flush()
    if eof:
      out_file.close()
      print("finished translating {}".format(hparams.out_file_list[test_idx]))
      if hparams.detok:
        _ = subprocess.getoutput(
        "python src/reversible_tokenize.py --detok < {0} > {1}".format( hparams.out_file_list[test_idx], hparams.out_file_list[test_idx] + ".detok"))

      test_idx += 1
      if not eop:
        out_file = open(hparams.out_file_list[test_idx], "w", encoding="utf-8")
    if eop:
      break    

if args.debug:
  forward_scores = []
  while not end_of_epoch:
    ((x_test, x_mask, x_len, x_count),
     (y_test, y_mask, y_len, y_count),
     batch_size, end_of_epoch), x_test_char, y_test_char = data.next_test(test_batch_size=hparams.batch_size, sort_by_x=True)
  
    num_sentences += batch_size
    logits = model.forward(x_test, x_mask, x_len, y_test[:,:-1,:], y_mask[:,:-1], y_len, y_test[:,1:,2], x_test_char, y_test_char)
    logits = logits.view(-1, hparams.target_rule_vocab_size+hparams.target_word_vocab_size)
    labels = y_test[:,1:,0].contiguous().view(-1)
    val_loss, val_acc, rule_loss, word_loss, eos_loss, rule_count, word_count, eos_count =  \
        get_performance(crit, logits, labels, hparams, sum_loss=False)
    print("train forward:", val_loss.data)
    print("train label:", labels.data)
    logit_score = []
    for i,l in enumerate(labels): logit_score.append(logits[i][l].data[0])
    print("train_logit", logit_score)
    #print("train_label", labels)
    forward_scores.append(val_loss.sum().data[0])
    # The normal, correct way:
    #hyps = model.translate(
    #      x_test, x_len, beam_size=args.beam_size, max_len=args.max_len)
    # For debugging:
    # model.debug_translate_batch(
    #   x_test, x_mask, x_pos_emb_indices, hparams.beam_size, hparams.max_len,
    #   y_test, y_mask, y_pos_emb_indices)
    # sys.exit(0)
  print("translate_score:", sum(scores))
  print("forward_score:", sum(forward_scores))
  exit(0)

if args.nbest:
  for h_list in hyps:
    for h in h_list:
      h_best_words = map(lambda wi: data.trg_i2w_list[0][wi],
                       filter(lambda wi: wi not in filts, h))
      if hparams.merge_bpe:
        line = ''.join(h_best_words)
        line = line.replace('▁', ' ')
      else:
        line = ' '.join(h_best_words)
      line = line.strip()
      out_file.write(line + '\n')
      out_file.flush()
    out_file.write('\n')
out_file.close()
