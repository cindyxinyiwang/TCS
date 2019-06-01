import numpy as np
import argparse
import time
import shutil
import gc
import random
import subprocess
import re

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils import DataUtil
from mult_data_utils import MultDataUtil
from hparams import *
from model import *
from utils import *
from transformer import *

parser = argparse.ArgumentParser(description="Neural MT")

parser.add_argument("--always_save", action="store_true", help="always_save")
parser.add_argument("--id_init_sep", action="store_true", help="init identity matrix")
parser.add_argument("--id_scale", type=float, default=0.01, help="[mlp|dot_prod|linear]")

parser.add_argument("--semb", type=str, default=None, help="[mlp|dot_prod|linear]")
parser.add_argument("--semb_num", type=int, default=1, help="number of categories for semb")
parser.add_argument("--dec_semb", action="store_true", help="load an existing model")
parser.add_argument("--query_base", action="store_true", help="load an existing model")
parser.add_argument("--semb_vsize", type=int, default=None, help="how many steps to write log")
parser.add_argument("--lan_code_rl", action="store_true", help="whether to set all unk words of rl to a reserved id")
parser.add_argument("--sample_rl", action="store_true", help="whether to set all unk words of rl to a reserved id")
parser.add_argument("--sep_char_proj", action="store_true", help="whether to have separate matrix for projecting char embedding")
parser.add_argument("--sep_relative_loc", action="store_true", help="whether to have separate transformer relative loc")
parser.add_argument("--residue", action="store_true", help="whether to set all unk words of rl to a reserved id")
parser.add_argument("--layer_norm", action="store_true", help="whether to set all unk words of rl to a reserved id")
parser.add_argument("--src_no_char", action="store_true", help="load an existing model")
parser.add_argument("--trg_no_char", action="store_true", help="load an existing model")
parser.add_argument("--char_gate", action="store_true", help="load an existing model")
parser.add_argument("--shuffle_train", action="store_true", help="load an existing model")
parser.add_argument("--lang_shuffle", action="store_true", help="load an existing model")
parser.add_argument("--ordered_char_dict", action="store_true", help="load an existing model")
parser.add_argument("--out_c_list", type=str, default=None, help="list of output channels for char cnn emb")
parser.add_argument("--k_list", type=str, default=None, help="list of kernel size for char cnn emb")
parser.add_argument("--highway", action="store_true", help="load an existing model")
parser.add_argument("--n", type=int, default=4, help="ngram n")
parser.add_argument("--single_n", action="store_true", help="ngram n")
parser.add_argument("--bpe_ngram", action="store_true", help="bpe ngram")
parser.add_argument("--uni", action="store_true", help="Gu Universal NMT")
parser.add_argument("--pretrained_src_emb_list", type=str, default=None, help="ngram n")
parser.add_argument("--pretrained_trg_emb", type=str, default=None, help="ngram n")

parser.add_argument("--load_model", action="store_true", help="load an existing model")
parser.add_argument("--reset_output_dir", action="store_true", help="delete output directory if it exists")
parser.add_argument("--output_dir", type=str, default="outputs", help="path to output directory")
parser.add_argument("--log_every", type=int, default=50, help="how many steps to write log")
parser.add_argument("--eval_every", type=int, default=500, help="how many steps to compute valid ppl")
parser.add_argument("--clean_mem_every", type=int, default=10, help="how many steps to clean memory")
parser.add_argument("--eval_bleu", action="store_true", help="if calculate BLEU score for dev set")
parser.add_argument("--beam_size", type=int, default=5, help="beam size for dev BLEU")
parser.add_argument("--poly_norm_m", type=float, default=1, help="beam size for dev BLEU")
parser.add_argument("--ppl_thresh", type=float, default=20, help="beam size for dev BLEU")
parser.add_argument("--max_trans_len", type=int, default=300, help="beam size for dev BLEU")
parser.add_argument("--merge_bpe", action="store_true", help="if calculate BLEU score for dev set")
parser.add_argument("--dev_zero", action="store_true", help="if eval at step 0")

parser.add_argument("--cuda", action="store_true", help="GPU or not")
parser.add_argument("--decode", action="store_true", help="whether to decode only")

parser.add_argument("--max_len", type=int, default=10000, help="maximum len considered on the target side")
parser.add_argument("--n_train_sents", type=int, default=None, help="max number of training sentences to load")

parser.add_argument("--d_word_vec", type=int, default=288, help="size of word and positional embeddings")
parser.add_argument("--d_char_vec", type=int, default=None, help="size of word and positional embeddings")
parser.add_argument("--d_model", type=int, default=288, help="size of hidden states")
parser.add_argument("--d_inner", type=int, default=512, help="hidden dim of position-wise ff")
parser.add_argument("--n_layers", type=int, default=1, help="number of lstm layers")
parser.add_argument("--n_heads", type=int, default=3, help="number of attention heads")
parser.add_argument("--d_k", type=int, default=64, help="size of attention head")
parser.add_argument("--d_v", type=int, default=64, help="size of attention head")
parser.add_argument("--pos_emb_size", type=int, default=None, help="size of trainable pos emb")

parser.add_argument("--data_path", type=str, default=None, help="path to all data")
parser.add_argument("--train_src_file_list", type=str, default=None, help="source train file")
parser.add_argument("--train_trg_file_list", type=str, default=None, help="target train file")
parser.add_argument("--dev_src_file_list", type=str, default=None, help="source valid file")
parser.add_argument("--dev_src_file", type=str, default=None, help="source valid file")
parser.add_argument("--dev_trg_file_list", type=str, default=None, help="target valid file")
parser.add_argument("--dev_trg_file", type=str, default=None, help="target valid file")
parser.add_argument("--dev_ref_file_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--dev_trg_ref", type=str, default=None, help="target valid file for reference")
parser.add_argument("--dev_file_idx_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--src_vocab_list", type=str, default=None, help="source vocab file")
parser.add_argument("--trg_vocab_list", type=str, default=None, help="target vocab file")
parser.add_argument("--test_src_file_list", type=str, default=None, help="source test file")
parser.add_argument("--test_src_file", type=str, default=None, help="source test file")
parser.add_argument("--test_trg_file_list", type=str, default=None, help="target test file")
parser.add_argument("--test_file_idx_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--test_trg_file", type=str, default=None, help="target test file")
parser.add_argument("--src_char_vocab_from", type=str, default=None, help="source char vocab file")
parser.add_argument("--src_char_vocab_size", type=str, default=None, help="source char vocab file")
parser.add_argument("--trg_char_vocab_from", type=str, default=None, help="source char vocab file")
parser.add_argument("--trg_char_vocab_size", type=str, default=None, help="source char vocab file")
parser.add_argument("--src_vocab_size", type=int, default=None, help="src vocab size")
parser.add_argument("--trg_vocab_size", type=int, default=None, help="trg vocab size")
# multi data util options
parser.add_argument("--lang_file", type=str, default=None, help="language code file")
parser.add_argument("--src_vocab", type=str, default=None, help="source vocab file")
parser.add_argument("--src_vocab_from", type=str, default=None, help="list of source vocab file")
parser.add_argument("--trg_vocab", type=str, default=None, help="source vocab file")

parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--valid_batch_size", type=int, default=20, help="batch_size")
parser.add_argument("--batcher", type=str, default="sent", help="sent|word. Batch either by number of words or number of sentences")
parser.add_argument("--n_train_steps", type=int, default=100000, help="n_train_steps")
parser.add_argument("--n_train_epochs", type=int, default=0, help="n_train_epochs")
parser.add_argument("--dropout", type=float, default=0., help="probability of dropping")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_dec", type=float, default=0.5, help="learning rate decay")
parser.add_argument("--lr_min", type=float, default=0.0001, help="min learning rate")
parser.add_argument("--lr_max", type=float, default=0.001, help="max learning rate")
parser.add_argument("--lr_dec_steps", type=int, default=0, help="cosine delay: learning rate decay steps")

parser.add_argument("--n_warm_ups", type=int, default=0, help="lr warm up steps")
parser.add_argument("--lr_schedule", action="store_true", help="whether to use transformer lr schedule")
parser.add_argument("--clip_grad", type=float, default=5., help="gradient clipping")
parser.add_argument("--l2_reg", type=float, default=0., help="L2 regularization")
parser.add_argument("--patience", type=int, default=-1, help="patience")
parser.add_argument("--eval_end_epoch", action="store_true", help="whether to reload the hparams")

parser.add_argument("--seed", type=int, default=19920206, help="random seed")

parser.add_argument("--init_range", type=float, default=0.1, help="L2 init range")
parser.add_argument("--init_type", type=str, default="uniform", help="uniform|xavier_uniform|xavier_normal|kaiming_uniform|kaiming_normal")

parser.add_argument("--share_emb_softmax", action="store_true", help="weight tieing")
parser.add_argument("--label_smoothing", type=float, default=None, help="label smooth")
parser.add_argument("--reset_hparams", action="store_true", help="whether to reload the hparams")

parser.add_argument("--char_ngram_n", type=int, default=0, help="use char_ngram embedding")
parser.add_argument("--max_char_vocab_size", type=int, default=None, help="char vocab size")

parser.add_argument("--char_input", type=str, default=None, help="[sum|cnn]")
parser.add_argument("--char_comb", type=str, default="add", help="[cat|add]")

parser.add_argument("--char_temp", type=float, default=None, help="temperature to combine word and char emb")

parser.add_argument("--pretrained_model", type=str, default=None, help="location of pretrained model")

parser.add_argument("--src_char_only", action="store_true", help="only use char emb on src")
parser.add_argument("--trg_char_only", action="store_true", help="only use char emb on trg")

parser.add_argument("--model_type", type=str, default="seq2seq", help="[seq2seq|transformer]")
parser.add_argument("--share_emb_and_softmax", action="store_true", help="only use char emb on trg")
parser.add_argument("--transformer_wdrop", action="store_true", help="whether to drop out word embedding of transformer")
parser.add_argument("--transformer_relative_pos", action="store_true", help="whether to use relative positional encoding of transformer")
parser.add_argument("--relative_pos_c", action="store_true", help="whether to use relative positional encoding of transformer")
parser.add_argument("--relative_pos_d", action="store_true", help="whether to use relative positional encoding of transformer")
parser.add_argument("--update_batch", type=int, default="1", help="for how many batches to call backward and optimizer update")
parser.add_argument("--layernorm_eps", type=float, default=1e-9, help="layernorm eps")

parser.add_argument("--sample_sep", type=float, default=0, help="probability of sample a swap for seprate position encoding")
parser.add_argument("--sep_step", type=int, default=0, help="step to separate")
parser.add_argument("--balance_idx", type=int, default=-1, help="step to separate")
parser.add_argument("--balance_ratio", type=int, default=1, help="balance ratio")
parser.add_argument("--sep_layer", type=str, default="", help="layers to sep loc")
parser.add_argument("--lan_pos_emb", action="store_true", help="max layer to sep loc")
parser.add_argument("--prob_rate_idx", type=int, default=-1, help="balance ratio")
parser.add_argument("--sep_head_weight", action="store_true", help="max layer to sep loc")
parser.add_argument("--max_loc_layer", type=int, default=1000, help="max layer to set location emb")
parser.add_argument("--sp_reg", action="store_true", help="max layer to sep loc")
parser.add_argument("--select_data", action="store_true", help="max layer to sep loc")
parser.add_argument("--cons_vocab", action="store_true", help="max layer to sep loc")
parser.add_argument("--sel", type=str, default="", help="selected data extension")
parser.add_argument("--sample_select", action="store_true", help="max layer to sep loc")
parser.add_argument("--sim", type=str, default="", help="selected data extension")
parser.add_argument("--sim_rank", type=str, default="", help="selected data extension")
parser.add_argument("--sample_select_tau_max", type=float, default=1., help="selected data extension")
parser.add_argument("--sample_select_tau_min", type=float, default=1., help="selected data extension")
parser.add_argument("--sample_select_tau_step", type=int, default=1., help="selected data extension")
parser.add_argument("--sample_swap", action="store_true", help="max layer to sep loc")
parser.add_argument("--sample_swap_p_max", type=float, default=1., help="selected data extension")
parser.add_argument("--sample_swap_p_min", type=float, default=0., help="selected data extension")
parser.add_argument("--sample_swap_p_step", type=int, default=1., help="selected data extension")

parser.add_argument("--sample_load", action="store_true", help="max layer to sep loc")
parser.add_argument("--sample_prob_list", type=str, default="", help="selected data extension")

parser.add_argument("--compute_ngram", action="store_true", help="max layer to sep loc")
parser.add_argument("--topk", type=int, default=10, help="top k languages to train")
parser.add_argument("--el", action="store_true", help="max layer to sep loc")
parser.add_argument("--detok", action="store_true", help="max layer to sep loc")
args = parser.parse_args()

if args.bpe_ngram: args.n = None

def eval(model, data, crit, step, hparams, eval_bleu=False,
         valid_batch_size=20, tr_logits=None):
  print("Eval at step {0}. valid_batch_size={1}".format(step, valid_batch_size))
  model.hparams.decode = True
  model.eval()
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_ppl, total_bleu = 0, 0
  ppl_list, bleu_list = [], []
  valid_bleu = None
  for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data.next_dev(dev_batch_size=valid_batch_size):
    # clear GPU memory
    gc.collect()

    # next batch
    # do this since you shift y_valid[:, 1:] and y_valid[:, :-1]
    y_count -= batch_size
    # word count
    valid_words += y_count

    logits = model.forward(
      x, x_mask, x_len, x_pos_emb_idxs,
      y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_char, y_char, file_idx=dev_file_index, step=step, x_rank=x_rank)
    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = y[:,1:].contiguous().view(-1)
    val_loss, val_acc = get_performance(crit, logits, labels, hparams)
    n_batches += batch_size
    valid_loss += val_loss.item()
    valid_acc += val_acc.item()
    if eof:
      val_ppl = np.exp(valid_loss / valid_words)
      print("ppl for dev {}".format(dev_file_index[0]))
      print("val_step={0:<6d}".format(step))
      print(" loss={0:<6.2f}".format(valid_loss / valid_words))
      print(" acc={0:<5.4f}".format(valid_acc / valid_words))
      print(" val_ppl={0:<.2f}".format(val_ppl))
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      total_ppl += val_ppl
      ppl_list.append(val_ppl)
    if eop:
      break
  # BLEU eval
  if eval_bleu:
    valid_hyp_file_list = [os.path.join(args.output_dir, "dev{}.trans_{}".format(i, step)) for i in hparams.dev_file_idx_list]
    out_file = open(valid_hyp_file_list[0], 'w', encoding='utf-8')
    if args.detok:
      valid_hyp_detok_file_list = [os.path.join(args.output_dir, "dev{}.trans_{}.detok".format(i, step)) for i in hparams.dev_file_idx_list]
    dev_idx = 0
    for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data.next_dev(dev_batch_size=1):
      if args.model_type == 'seq2seq':
        hs = model.translate(
                x, x_mask, beam_size=args.beam_size, max_len=args.max_trans_len, poly_norm_m=args.poly_norm_m, x_train_char=x_char, y_train_char=y_char, file_idx=dev_file_index, step=step, x_rank=x_rank)
      elif args.model_type == 'transformer': 
        hs = model.translate(
                x, x_mask, x_pos_emb_idxs, x_char_sparse_batch=x_char, beam_size=args.beam_size, max_len=args.max_trans_len, poly_norm_m=args.poly_norm_m, file_idx=dev_file_index, step=step)
      for h in hs:
        h_best_words = map(lambda wi: data.trg_i2w[wi],
                         filter(lambda wi: wi not in [hparams.bos_id, hparams.eos_id], h))
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
        if args.detok:
          _ = subprocess.getoutput(
          "python src/reversible_tokenize.py --detok < {0} > {1}".format(valid_hyp_file_list[dev_idx], valid_hyp_detok_file_list[dev_idx]))

        ref_file = hparams.dev_ref_file_list[dev_idx]
        if args.detok:
          bleu_str = subprocess.getoutput(
            "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_detok_file_list[dev_idx]))
        else:
          bleu_str = subprocess.getoutput(
            "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_file_list[dev_idx]))
        print("bleu for dev {}".format(dev_idx))
        print("{}".format(bleu_str))
        bleu_str = bleu_str.split('\n')[-1].strip()
        reg = re.compile("BLEU = ([^,]*).*")
        try:
          valid_bleu = float(reg.match(bleu_str).group(1))
        except:
          valid_bleu = 0.
        print(" val_bleu={0:<.2f}".format(valid_bleu))
        total_bleu += valid_bleu
        dev_idx += 1
        bleu_list.append(valid_bleu)
        if not eop:
          out_file = open(valid_hyp_file_list[dev_idx], "w", encoding="utf-8")
      if eop:
        break
  model.hparams.decode = False
  model.train()
  return total_ppl, total_bleu, ppl_list, bleu_list

def train():
  if args.load_model and (not args.reset_hparams):
    print("load hparams..")
    if len(args.dev_src_file_list.split(",")) > 1:
      hparams_file_name = os.path.join(args.output_dir, "dev0/hparams.pt")
    else:
      hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
    hparams.load_model = args.load_model
    hparams.n_train_steps = args.n_train_steps
  else:
    hparams = HParams(
      decode=args.decode,
      data_path=args.data_path,
      train_src_file_list=args.train_src_file_list,
      train_trg_file_list=args.train_trg_file_list,
      dev_src_file=args.dev_src_file,
      dev_trg_file=args.dev_trg_file,
      dev_src_file_list=args.dev_src_file_list,
      dev_trg_file_list=args.dev_trg_file_list,
      dev_ref_file_list=args.dev_ref_file_list,
      dev_file_idx_list=args.dev_file_idx_list,
      lang_file=args.lang_file,
      src_vocab=args.src_vocab,
      trg_vocab=args.trg_vocab,
      src_vocab_list=args.src_vocab_list,
      trg_vocab_list=args.trg_vocab_list,
      src_vocab_size=args.src_vocab_size,
      trg_vocab_size=args.trg_vocab_size,
      max_len=args.max_len,
      n_train_sents=args.n_train_sents,
      cuda=args.cuda,
      d_word_vec=args.d_word_vec,
      d_model=args.d_model,
      d_inner=args.d_inner,
      n_layers=args.n_layers,
      batch_size=args.batch_size,
      batcher=args.batcher,
      n_train_steps=args.n_train_steps,
      dropout=args.dropout,
      lr=args.lr,
      lr_dec=args.lr_dec,
      l2_reg=args.l2_reg,
      init_type=args.init_type,
      init_range=args.init_range,
      share_emb_softmax=args.share_emb_softmax,
      label_smoothing=args.label_smoothing,
      n_heads=args.n_heads,
      d_k=args.d_k,
      d_v=args.d_v,
      merge_bpe=args.merge_bpe,
      load_model=args.load_model,
      char_ngram_n=args.char_ngram_n,
      max_char_vocab_size=args.max_char_vocab_size,
      char_input=args.char_input,
      char_comb=args.char_comb,
      char_temp=args.char_temp,
      src_char_vocab_from=args.src_char_vocab_from,
      src_char_vocab_size=args.src_char_vocab_size,
      trg_char_vocab_from=args.trg_char_vocab_from,
      trg_char_vocab_size=args.trg_char_vocab_size,
      src_char_only=args.src_char_only,
      trg_char_only=args.trg_char_only,
      semb=args.semb,
      semb_num=args.semb_num,
      dec_semb=args.dec_semb,
      semb_vsize=args.semb_vsize,
      lan_code_rl=args.lan_code_rl,
      sample_rl=args.sample_rl,
      sep_char_proj=args.sep_char_proj,
      sep_relative_loc=args.sep_relative_loc,
      query_base=args.query_base,
      residue=args.residue,
      layer_norm=args.layer_norm,
      src_no_char=args.src_no_char,
      trg_no_char=args.trg_no_char,
      char_gate=args.char_gate,
      shuffle_train=args.shuffle_train,
      lang_shuffle=args.lang_shuffle,
      ordered_char_dict=args.ordered_char_dict,
      out_c_list=args.out_c_list,
      k_list=args.k_list,
      d_char_vec=args.d_char_vec,
      highway=args.highway,
      n=args.n,
      single_n=args.single_n,
      bpe_ngram=args.bpe_ngram,
      uni=args.uni,
      pretrained_src_emb_list=args.pretrained_src_emb_list,
      pretrained_trg_emb=args.pretrained_trg_emb,
      pos_emb_size=args.pos_emb_size,
      lr_schedule=args.lr_schedule,
      lr_max=args.lr_max,
      lr_min=args.lr_min,
      lr_dec_steps=args.lr_dec_steps,
      n_warm_ups=args.n_warm_ups,
      model_type=args.model_type,
      transformer_wdrop=args.transformer_wdrop,
      transformer_relative_pos=args.transformer_relative_pos,
      relative_pos_c=args.relative_pos_c,
      relative_pos_d=args.relative_pos_d,
      sample_sep=args.sample_sep,
      sep_step=args.sep_step,
      balance_idx=args.balance_idx,
      balance_ratio=args.balance_ratio,
      sep_layer=args.sep_layer,
      lan_pos_emb=args.lan_pos_emb,
      prob_rate_idx=args.prob_rate_idx,
      sep_head_weight=args.sep_head_weight,
      max_loc_layer=args.max_loc_layer,
      select_data=args.select_data,
      cons_vocab=args.cons_vocab,
      sel=args.sel,
      sample_select=args.sample_select,
      sim=args.sim,
      sim_rank=args.sim_rank,
      sample_select_tau_max=args.sample_select_tau_max,
      sample_select_tau_min=args.sample_select_tau_min,
      sample_select_tau_step=args.sample_select_tau_step,
      sample_swap=args.sample_swap,
      sample_swap_p_max=args.sample_swap_p_max,
      sample_swap_p_min=args.sample_swap_p_min,
      sample_swap_p_step=args.sample_swap_p_step,
      sample_load=args.sample_load,
      sample_prob_list=args.sample_prob_list,
      compute_ngram=args.compute_ngram,
      update_batch=args.update_batch,
      topk=args.topk,
      el=args.el,
      detok=args.detok,
    )
  # build or load model
  print("-" * 80)
  print("Creating model")
  if args.load_model:
    data = MultDataUtil(hparams=hparams)
    if len(hparams.dev_src_file_list) > 1:
      model_file_name = os.path.join(args.output_dir, "dev0/model.pt")
    else:
      model_file_name = os.path.join(args.output_dir, "model.pt")
    print("Loading model from '{0}'".format(model_file_name))
    model = torch.load(model_file_name)
    if not hasattr(model, 'data'):
      model.data = data
    if not hasattr(model.hparams, 'transformer_wdrop'):
      model.hparams.transformer_wdrop = False

    if len(hparams.dev_src_file_list) > 1:
      optim_file_name = os.path.join(args.output_dir, "dev0/optimizer.pt")
    else:
      optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optimizer from {}".format(optim_file_name))
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    #optim = torch.optim.Adam(trainable_params, lr=hparams.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=hparams.l2_reg)
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr, weight_decay=hparams.l2_reg)
    optimizer_state = torch.load(optim_file_name)
    optim.load_state_dict(optimizer_state)

    if len(hparams.dev_src_file_list) > 1:
      extra_file_name = os.path.join(args.output_dir, "dev0/extra.pt")
    else:
      extra_file_name = os.path.join(args.output_dir, "extra.pt")
    step, best_val_ppl, best_val_bleu, cur_attempt, lr = torch.load(extra_file_name)
  else:
    if args.pretrained_model:
      model_name = os.path.join(args.pretrained_model, "model.pt")
      print("Loading model from '{0}'".format(model_name))
      model = torch.load(model_name)
      #if not hasattr(model, 'data'):
      #  model.data = data
      #if not hasattr(model, 'char_ngram_n'):
      #  model.hparams.char_ngram_n = 0
      #if not hasattr(model, 'char_input'):
      #  model.hparams.char_input = None
      print("load hparams..")
      hparams_file_name = os.path.join(args.pretrained_model, "hparams.pt")
      reload_hparams = torch.load(hparams_file_name)
      reload_hparams.train_src_file_list = hparams.train_src_file_list
      reload_hparams.train_trg_file_list = hparams.train_trg_file_list
      reload_hparams.dropout = hparams.dropout
      reload_hparams.lr_dec = hparams.lr_dec
      hparams = reload_hparams
      #hparams.src_vocab_list = reload_hparams.src_vocab_list 
      #hparams.src_vocab_size = reload_hparams.src_vocab_size 
      #hparams.trg_vocab_list = reload_hparams.trg_vocab_list 
      #hparams.trg_vocab_size = reload_hparams.trg_vocab_size 
      #hparams.src_char_vocab_from = reload_hparams.src_char_vocab_from 
      #hparams.src_char_vocab_size = reload_hparams.src_char_vocab_size 
      #hparams.trg_char_vocab_from = reload_hparams.trg_char_vocab_from 
      #hparams.trg_char_vocab_size = reload_hparams.trg_char_vocab_size
      #print(reload_hparams.src_char_vocab_from)
      #print(reload_hparams.src_char_vocab_size)
      data = MultDataUtil(hparams=hparams)
      model.data = data
    else:
      data = MultDataUtil(hparams=hparams)
      if args.model_type == 'seq2seq':
        model = Seq2Seq(hparams=hparams, data=data)
      elif args.model_type == 'transformer':
        model = Transformer(hparams=hparams, data=data)
      else:
        print("Model {} not implemented".format(args.model_type))
        exit(0)
      if args.init_type == "uniform" and not hparams.model_type == "transformer":
        print("initialize uniform with range {}".format(args.init_range))
        for p in model.parameters():
          p.data.uniform_(-args.init_range, args.init_range)
      if args.id_init_sep and args.semb and args.sep_char_proj:
        print("initialize char proj as identity matrix")
        for s in model.encoder.char_emb.sep_proj_list:
          d = s.weight.data.size(0)
          s.weight.data.copy_(torch.eye(d) + args.id_scale*torch.diagflat(torch.ones(d).normal_(0,1)))
      if args.uni:
        print("initialize A as identity matrix")
        nn.init.eye(model.encoder.shared_emb.A.weight)
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr, weight_decay=hparams.l2_reg)
    #optim = torch.optim.Adam(trainable_params)
    step = 0
    best_val_ppl = [None for _ in range(len(model.hparams.dev_src_file_list))]
    best_val_bleu = [None for _ in range(len(model.hparams.dev_src_file_list))]
    cur_attempt = 0
    lr = hparams.lr

  if args.cuda:
    model = model.cuda()

  if args.reset_hparams:
    lr = args.lr
  crit = get_criterion(hparams)
  trainable_params = [
    p for p in model.parameters() if p.requires_grad]
  num_params = count_params(trainable_params)
  print("Model has {0} params".format(num_params))

  print("-" * 80)
  print("start training...")
  start_time = log_start_time = time.time()
  target_words, total_loss, total_corrects = 0, 0, 0
  target_rules, target_total, target_eos = 0, 0, 0
  total_word_loss, total_rule_loss, total_eos_loss = 0, 0, 0
  model.train()
  #i = 0
  epoch = 0
  dev_zero = args.dev_zero
  baseline_loss = None
  tr_loss, update_batch_size = None, 0
  hparams.new_lan_warm = False
  s0_trainable_params = []
  #get_grad_cos_all(model, data, crit)
  #data.update_base_prob_list()
  for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank) in data.next_train():
    step += 1
    target_words += (y_count - batch_size)
    logits = model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_train_char_sparse, y_train_char_sparse, file_idx=file_idx, step=step, x_rank=x_rank)
    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = y_train[:,1:].contiguous().view(-1)
      
    cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, hparams)
    total_loss += cur_tr_loss.item()
    total_corrects += cur_tr_acc.item()
    cur_tr_loss.div_(batch_size * hparams.update_batch)

    cur_tr_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

    if file_idx[0] == args.prob_rate_idx-1:
      baseline_loss = cur_tr_loss.item() / (y_count - batch_size)
    if file_idx[0] == args.prob_rate_idx: 
      if step  % args.log_every == 0:
        print("cur loss: {}".format((cur_tr_loss.item() / (y_count - batch_size))) )
      weight = 0.5 * min(1, step / 30000)
      cur_tr_loss = cur_tr_loss * weight
      if step  % args.log_every == 0:
        print("baseline: {}, weight: {}".format(baseline_loss, weight))
    
    if tr_loss is None:
      tr_loss = cur_tr_loss.item()
    else:
      tr_loss = tr_loss + cur_tr_loss.item()
    #update_batch_size += batch_size
    if dev_zero:
      dev_zero = False
      #based_on_bleu = args.eval_bleu and best_val_ppl <= args.ppl_thresh
      based_on_bleu = args.eval_bleu
      val_ppl, val_bleu = eval(model, data, crit, step, hparams, eval_bleu=based_on_bleu, valid_batch_size=args.valid_batch_size, tr_logits=logits)	
      if based_on_bleu:
        if best_val_bleu <= val_bleu:
          save = True 
          best_val_bleu = val_bleu
          cur_attempt = 0
        else:
          save = False
          cur_attempt += 1
      else:
      	if best_val_ppl >= val_ppl:
          save = True
          best_val_ppl = val_ppl
          cur_attempt = 0 
      	else:
          save = False
          cur_attempt += 1
      if save or args.always_save:
      	save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, lr], 
      		             model, optim, hparams, args.output_dir)
      elif not args.lr_schedule and step >= hparams.n_warm_ups:
        lr = lr * args.lr_dec
        set_lr(optim, lr)
      # reset counter after eval
      log_start_time = time.time()
      target_words = total_corrects = total_loss = 0
      target_rules = target_total = target_eos = 0
      total_word_loss = total_rule_loss = total_eos_loss = 0
    if step % args.update_batch == 0:
      # set learning rate
      if args.lr_schedule:
        s = step / args.update_batch + 1
        lr = pow(hparams.d_model, -0.5) * min(
          pow(s, -0.5), s * pow(hparams.n_warm_ups, -1.5))
        set_lr(optim, lr)
      elif step / args.update_batch < hparams.n_warm_ups:
        base_lr = hparams.lr
        base_lr = base_lr * (step / args.update_batch + 1) / hparams.n_warm_ups
        set_lr(optim, base_lr)
        lr = base_lr
      elif args.lr_dec_steps > 0:
        s = (step / args.update_batch) % args.lr_dec_steps
        lr = args.lr_min + 0.5*(args.lr_max-args.lr_min)*(1+np.cos(s*np.pi/args.lr_dec_steps))
        set_lr(optim, lr)
      if args.sp_reg and s0_trainable_params:
        total_norm = 0
        assert len(s0_trainable_params) == len(trainable_params)
        for p_s, p in zip(s0_trainable_params, trainable_params):
          param_norm = (p - p_s).norm(2)
          total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        tr_loss = tr_loss + total_norm
       
      #grad_norm = grad_clip(trainable_params, grad_bound=args.clip_grad)
      if np.isnan(grad_norm):
          print("WARNING: gradient is nan!, skipping batch")
          print("x:", x_train.data)
          print("y:", y_train.data)
          print("logits:", logits.data)
          ##print("x_emb grad:", model.encoder.word_emb.weight.grad)
          #print("y_emb grad:", model.decoder.word_emb.weight.grad)
          #print("y_emb grad:", model.decoder.word_emb.weight.grad.size())
          #print("y layer norm grad:", model.decoder.layer_stack[0].y_attn.layer_norm.scale.grad.data)
          #print("y layer norm grad:", model.decoder.layer_stack[0].y_attn.layer_norm.offset.grad.data)
          #print("decoder layer 0")
          #for p in model.decoder.layer_stack[0].y_attn.parameters():
          #  if not p.requires_grad: continue
          #  print(p.data)
          #  print(p.grad)
          #  print(p.grad.size())
          exit(0)
      optim.step()
      optim.zero_grad()
      tr_loss = None
      update_batch_size = 0
    # clean up GPU memory
    if step % args.clean_mem_every == 0:
      gc.collect()
    if eop: 
      epoch += 1
      #data.update_base_prob_list()
    #if eof and file_idx[0] == 0:
    #  get_grad_cos_all(model, data, crit)
    if (step / args.update_batch) % args.log_every == 0:
      curr_time = time.time()
      since_start = (curr_time - start_time) / 60.0
      elapsed = (curr_time - log_start_time) / 60.0
      log_string = "ep={0:<3d}".format(epoch)
      log_string += " steps={0:<6.2f}".format((step / args.update_batch) / 1000)
      log_string += " lr={0:<9.7f}".format(lr)
      log_string += " loss={0:<7.2f}".format(cur_tr_loss.item())
      log_string += " |g|={0:<5.2f}".format(grad_norm)

      log_string += " ppl={0:<8.2f}".format(np.exp(total_loss / target_words))
      log_string += " acc={0:<5.4f}".format(total_corrects / target_words)

      log_string += " wpm(k)={0:<5.2f}".format(target_words / (1000 * elapsed))
      log_string += " time(min)={0:<5.2f}".format(since_start)
      print(log_string)
    if args.eval_end_epoch:
      if eop:
        eval_now = True
      else:
        eval_now = False
    elif (step / args.update_batch) % args.eval_every == 0:
      eval_now = True
    else:
      eval_now = False 
    if eval_now:
      based_on_bleu = args.eval_bleu and best_val_ppl[0] is not None and best_val_ppl[0] <= args.ppl_thresh
      if args.dev_zero: based_on_bleu = True
      with torch.no_grad():
        val_ppl, val_bleu, ppl_list, bleu_list = eval(model, data, crit, step, hparams, eval_bleu=based_on_bleu, valid_batch_size=args.valid_batch_size, tr_logits=logits)	
      for i in range(len(ppl_list)):
        if based_on_bleu:
          if best_val_bleu[i] is None or best_val_bleu[i] <= bleu_list[i]:
            save = True 
            best_val_bleu[i] = bleu_list[i]
            cur_attempt = 0
          else:
            save = False
            cur_attempt += 1
        else:
       	  if best_val_ppl[i] is None or best_val_ppl[i] >= ppl_list[i]:
            save = True
            best_val_ppl[i] = ppl_list[i]
            cur_attempt = 0 
          else:
            save = False
            cur_attempt += 1
        if save or args.always_save:
          if len(ppl_list) > 1:
            save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, lr], model, optim, hparams, args.output_dir + "dev{}".format(i))
          else:
            save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, lr], model, optim, hparams, args.output_dir)
          if args.sp_reg:
            s0_trainable_params = []
            for p in model.parameters(): 
              if p.requires_grad: 
                s0_trainable_params.append(p.detach().clone())
        elif not args.lr_schedule and step >= hparams.n_warm_ups:
          lr = lr * args.lr_dec
          set_lr(optim, lr)
      # reset counter after eval
      log_start_time = time.time()
      target_words = total_corrects = total_loss = 0
      target_rules = target_total = target_eos = 0
      total_word_loss = total_rule_loss = total_eos_loss = 0
    if args.patience >= 0:
      if cur_attempt > args.patience: break
    elif args.n_train_epochs > 0:
      if epoch >= args.n_train_epochs: break
    else:
      if step > args.n_train_steps: break

def main():
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  dev_src_file_list = args.dev_src_file_list.split(",")
  if not os.path.isdir(args.output_dir):
    print("-" * 80)
    print("Path {} does not exist. Creating.".format(args.output_dir))
    os.makedirs(args.output_dir)
  elif args.reset_output_dir:
    print("-" * 80)
    print("Path {} exists. Remove and remake.".format(args.output_dir))
    shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
  if len(dev_src_file_list) > 1:
    for i in range(len(dev_src_file_list)):
      if not os.path.isdir(args.output_dir + "dev{}".format(i)):
        os.makedirs(args.output_dir + "dev{}".format(i))
      #else:
      #  shutil.rmtree(args.output_dir + "dev{}".format(i))
      #  os.makedirs(args.output_dir + "dev{}".format(i))
  print("-" * 80)
  log_file = os.path.join(args.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)
  train()

if __name__ == "__main__":
  main()
