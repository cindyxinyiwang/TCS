import argparse
import pickle as pkl
from scipy.special import logsumexp

from model import *
from transformer import *
from utils import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from mult_data_utils import MultDataUtil
from select_sent import get_lan_order

vocab_size = 8000
base_lan = "bel"
#lan_lists = ["ukr", "rus", "bul", "mkd", "kaz", "mon"]
#lan_lists = ["aze", "tur", "rus", "por", "ces"]
#lan_lists = ["tur", "ind", "msa", "epo", "sqi", "swe", "dan"]
#lan_lists = ["por", "spa", "ita", "fra", "ron", "epo"]
#lan_lists = ["ces", "slv", "hrv", "bos", "srp"]
cuda = True

data_dir = "/projects/tir3/users/xinyiw1/data_iwslt/" 

def sent_vocab_overlap():
  base_lans = ["aze", "bel", "glg", "slk"]
  for base_lan in base_lans:
    base_lan_vocab_file = "data/{}_eng/ted-train.mtok.{}.char4vocab".format(base_lan, base_lan)
    base_lan_vocab = []
    with open(base_lan_vocab_file, 'r') as myfile:
      for line in myfile: base_lan_vocab.append(line.strip())
    base_lan_vocab = set(base_lan_vocab)

    lan_lists = [l.strip() for l in open("langs.txt", 'r').readlines()]
    lans = []
    for l in lan_lists:
      if l != base_lan: lans.append(l)
    lan_lists = lans

    for lan in lan_lists:
      lan_file = "data/{}_eng/ted-train.mtok.{}".format(lan, lan)
      out_file = "lmll/ted-train.mtok.{}.{}-vocab".format(lan, base_lan)
      out_file = open(out_file, 'w')
      with open(lan_file, 'r') as myfile:
        for line in myfile:
          words = line.split()
          total_count, match_count = 0, 0
          for w in words:
            for i in range(len(w)):
              for j in range(i+1, min(i+4, len(w))+1):
                char = w[i:j]
                total_count += 1
                if char in base_lan_vocab: match_count += 1
          out_file.write("{}\n".format(match_count / total_count))
         
def prob_by_vocab_overlap_sent():
  base_lans = ["aze"]
  #ts = [0.01, 0.05, 0.1, 0.1]
  #argmaxs = [False, False, False, True]
  ts = [0.1]
  argmaxs = [True]
  for base_lan in base_lans:
    for t, argmax in zip(ts, argmaxs):
      trg2srcs = {}
      lan_lists = [l.strip() for l in open("langs.txt", 'r').readlines()]
      lans = []
      for l in lan_lists:
        if l != base_lan: lans.append(l)
      lan_lists = lans

      out_probs = []
      for i, lan in enumerate(lan_lists):
        lm_file = "lmll/ted-train.mtok.{}.{}-vocab".format(lan, base_lan)
        lm_score = [float(l) for l in open(lm_file, 'r').readlines()]

        trg_file = "data/{}_eng/ted-train.mtok.spm8000.eng".format(lan)
        trg_sents = open(trg_file, 'r').readlines()
        out_probs.append([0 for _ in range(len(trg_sents))])
        line = 0
        for j, trg in enumerate(trg_sents):
          if trg not in trg2srcs: trg2srcs[trg] = []
          trg2srcs[trg].append([i, line, lm_score[j]])
          line += 1
      print("eng size: {}".format(len(trg2srcs)))
      for trg, src_list in trg2srcs.items():
        if argmax:
          max_score = 0
          for s in src_list:
            max_score = max(s[2], max_score)
          for s in src_list:
            if s[2] == max_score:
              out_probs[s[0]][s[1]] = 1
            else:
              out_probs[s[0]][s[1]] = 0
        else:
          sum_score = 0
          log_score = []
          for s in src_list:
            #s[2] = np.exp(-s[2] / t)
            #sum_score += s[2]
            s[2] = s[2] / t
            log_score.append(s[2])
          sum_score = logsumexp(log_score)
          for s in src_list:
            #s[2] = s[2] / sum_score
            s[2] = np.exp(s[2] - sum_score)
            out_probs[s[0]][s[1]] = s[2]

      for i, lan in enumerate(lan_lists):
        if argmax:
          out = open("data/{}_eng/ted-train.mtok.{}.prob-vocab-sent-{}-am".format(lan, lan, base_lan), "w")
        else:
          out = open("data/{}_eng/ted-train.mtok.{}.prob-vocab-sent-{}-t{}".format(lan, lan, base_lan, t), "w")
        #out = open(data_dir + "{}_en/ted-train.mtok.{}.prob-rank-{}-t{}-k{}-el".format(lan, lan, base_lan, t, k), "w")
        for p in out_probs[i]:
          out.write("{}\n".format(p))
        out.close()
      if argmax:
        out = open("data/{}_eng/ted-train.mtok.{}.prob-vocab-sent-{}-am".format(base_lan, base_lan, base_lan), "w")
      else:
        out = open("data/{}_eng/ted-train.mtok.{}.prob-vocab-sent-{}-t{}".format(base_lan, base_lan, base_lan, t), "w")
      #out = open(data_dir + "{}_en/ted-train.mtok.{}.prob-rank-{}-t{}-k{}".format(base_lan, base_lan, base_lan, t, k), "w")
      base_lines = len(open("data/{}_eng/ted-train.mtok.spm8000.eng".format(base_lan)).readlines())
      #base_lines = len(open(data_dir + "{}_en/ted-train.mtok.spm8000.en".format(base_lan)).readlines())
      for i in range(base_lines):
        out.write("{}\n".format(1))
      out.close()


def prob_by_rank():
  trg2srcs = {}
  t = 1
  k = 58
  #k = 10
  #k = 29
  #out_lans = "lang_azeall_k29.txt"
  out_lans = ""
  # exclude low resource lan
  el = True
  #el = False
  lan_order, _ = get_lan_order(base_lan, lan_dist_file="ted-train-vocab.mtok.sim-ngram.graph")
  #lan_order, _ = get_lan_order(base_lan, lan_dist_file="ted-iwslt-vocab.mtok.sim-ngram.graph")
  #lan_order = lan_order[-k:-1]
  if el: 
    lan_order = lan_order[-k:]
  else:
    lan_order = lan_order[-k:-1]
  lan_lists = [kv[0] for kv in lan_order]
  # ngrams
  sim_rank = [kv[1]/100 for kv in lan_order]
  # spm8000
  #sim_rank = [kv[1]/10 for kv in lan_order]
  print(lan_lists)
  print(sim_rank)
  if out_lans:
    out_lans = open(out_lans, 'w')
    for l in lan_lists[::-1]:
      out_lans.write(l + "\n")
    out_lans.close()
  # aze
  #sim_rank = [48.36, 26.5, 25.12, 23.94, 23.89, 23.78, 23.31]
  # bel
  #sim_rank = [34.27, 32.09, 25.73, 23.99, 19.41, 16.84]
  # glg
  #sim_rank = [66.02, 72.04, 52.27, 45.33, 45.11, 39.85]
  #sim_rank = [72.04, 66.02, 52.27, 45.33, 45.11, 39.85]
  # slk
  #sim_rank = [63.31, 42.56, 40.76, 39.41, 36.73]
  sim_rank = [i/t for i in sim_rank]
  out_probs = []
  for i, lan in enumerate(lan_lists):
    trg_file = "data/{}_eng/ted-train.mtok.spm8000.eng".format(lan)
    #trg_file = data_dir + "{}_en/ted-train.mtok.spm8000.en".format(lan)
    trg_sents = open(trg_file, 'r').readlines()
    out_probs.append([0 for _ in range(len(trg_sents))])
    line = 0
    for trg in trg_sents:
      if trg not in trg2srcs: trg2srcs[trg] = []
      trg2srcs[trg].append([i, line, sim_rank[i]])
      line += 1
  print("eng size: {}".format(len(trg2srcs)))
  for trg, src_list in trg2srcs.items():
    sum_score = 0
    for s in src_list:
      s[2] = np.exp(s[2])
      sum_score += s[2]
    for s in src_list:
      s[2] = s[2] / sum_score
      out_probs[s[0]][s[1]] = s[2]

  for i, lan in enumerate(lan_lists):
    if el:
      out = open("data/{}_eng/ted-train.mtok.{}.prob-rank-{}-t{}-k{}-el".format(lan, lan, base_lan, t, k), "w")
    else:
      out = open("data/{}_eng/ted-train.mtok.{}.prob-rank-{}-t{}-k{}".format(lan, lan, base_lan, t, k), "w")
      #out = open(data_dir + "{}_en/ted-train.mtok.{}.prob-rank-{}-t{}-k{}-el".format(lan, lan, base_lan, t, k), "w")
    for p in out_probs[i]:
      out.write("{}\n".format(p))
    out.close()
  if el:
    out = open("data/{}_eng/ted-train.mtok.{}.prob-rank-{}-t{}-k{}-el".format(base_lan, base_lan, base_lan, t, k), "w")
  else:
    out = open("data/{}_eng/ted-train.mtok.{}.prob-rank-{}-t{}-k{}".format(base_lan, base_lan, base_lan, t, k), "w")
    #out = open(data_dir + "{}_en/ted-train.mtok.{}.prob-rank-{}-t{}-k{}-el".format(base_lan, base_lan, base_lan, t, k), "w")
  base_lines = len(open("data/{}_eng/ted-train.mtok.spm8000.eng".format(base_lan)).readlines())
  #base_lines = len(open(data_dir + "{}_en/ted-train.mtok.spm8000.en".format(base_lan)).readlines())
  for i in range(base_lines):
    out.write("{}\n".format(1))
  out.close()

def prob_by_lm():
  trg2srcs = {}
  t = 0.5
  lan_lists = [l.strip() for l in open("langs.txt", 'r').readlines()]
  
  lans = []
  for l in lan_lists:
    if l != base_lan: lans.append(l)
  lan_lists = lans

  argmax = False
  out_probs = []
  for i, lan in enumerate(lan_lists):
    lm_file = "data/{}_eng/ted-train.mtok.{}.{}-lmll".format(lan, lan, base_lan)
    lm_score = [float(l) for l in open(lm_file, 'r').readlines()]

    trg_file = "data/{}_eng/ted-train.mtok.spm8000.eng".format(lan)
    trg_sents = open(trg_file, 'r').readlines()
    out_probs.append([0 for _ in range(len(trg_sents))])
    line = 0
    for j, trg in enumerate(trg_sents):
      if trg not in trg2srcs: trg2srcs[trg] = []
      trg2srcs[trg].append([i, line, lm_score[j]])
      line += 1
  print("eng size: {}".format(len(trg2srcs)))
  for trg, src_list in trg2srcs.items():
    if argmax:
      max_score = 100000
      for s in src_list:
        max_score = min(s[2], max_score)
      for s in src_list:
        if s[2] == max_score:
          out_probs[s[0]][s[1]] = 1
        else:
          out_probs[s[0]][s[1]] = 0
    else:
      sum_score = 0
      for s in src_list:
        s[2] = np.exp(-s[2] / t)
        sum_score += s[2]
      for s in src_list:
        s[2] = s[2] / sum_score
        out_probs[s[0]][s[1]] = s[2]

  for i, lan in enumerate(lan_lists):
    if argmax:
      out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-{}-am".format(lan, lan, base_lan), "w")
    else:
      out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-{}-t{}".format(lan, lan, base_lan, t), "w")
    #out = open(data_dir + "{}_en/ted-train.mtok.{}.prob-rank-{}-t{}-k{}-el".format(lan, lan, base_lan, t, k), "w")
    for p in out_probs[i]:
      out.write("{}\n".format(p))
    out.close()
  if argmax:
    out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-{}-am".format(base_lan, base_lan, base_lan), "w")
  else:
    out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-{}-t{}".format(base_lan, base_lan, base_lan, t), "w")
  base_lines = len(open("data/{}_eng/ted-train.mtok.spm8000.eng".format(base_lan)).readlines())
  #base_lines = len(open(data_dir + "{}_en/ted-train.mtok.spm8000.en".format(base_lan)).readlines())
  for i in range(base_lines):
    out.write("{}\n".format(1))
  out.close()


def prob_by_classify():
  trg2srcs = {}
  out_probs = []
  for i, lan in enumerate(lan_lists):
    trg_file = "data/{}_eng/ted-train.mtok.spm8000.eng".format(lan)
    trg_sents = open(trg_file, 'r').readlines()
    sim_file = "data/{}_eng/ted-train.mtok.{}.{}".format(lan, lan, "sim-ngram_v1")
    sim_score = []
    with open(sim_file) as myfile:
      for line in myfile:
        if i < 1:
          sim_score.append(20)
          #sim_score.append(float(line.strip()))
        else:
          sim_score.append(float(line.strip()))
    out_probs.append([0 for _ in range(len(trg_sents))])
    line = 0
    for trg, s in zip(trg_sents, sim_score):
      if trg not in trg2srcs: trg2srcs[trg] = []
      trg2srcs[trg].append([i, line, s])
      line += 1
  print("eng size: {}".format(len(trg2srcs)))
  for trg, src_list in trg2srcs.items():
    sum_score = 0
    for s in src_list:
      s[2] = np.exp(s[2])
      sum_score += s[2]
    for s in src_list:
      s[2] = s[2] / sum_score
      out_probs[s[0]][s[1]] = s[2]

  for i, lan in enumerate(lan_lists):
    out = open("data/{}_eng/ted-train.mtok.{}.prob-sim-ngram".format(lan, lan), "w")
    for p in out_probs[i]:
      out.write("{}\n".format(p))
  #out = open("data/{}_eng/ted-train.mtok.{}.prob".format(base_lan, base_lan), "w")

def sim_by_model(model_dir):
  model_file_name = os.path.join(model_dir, "model.pt")
  if not cuda:
    model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
  else:
    model = torch.load(model_file_name)
  model.eval()
  
  model.hparams.shuffle_train = False
  model.hparams.batcher = "sent"
  model.hparams.batch_size =  1
  model.hparams.cuda = cuda
  model.hparams.sample_select = False 
  model.hparams.sep_char_proj = False 
  crit = get_criterion(model.hparams)
  for lan in lan_lists:
    model.hparams.lang_file = "lang_{}.txt".format(lan)
    data = MultDataUtil(model.hparams, shuffle=False)
    out = open("data/{}_eng/ted-train.mtok.{}.sim-nmt".format(lan, lan), "w")
 
    step = 0
    for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, dev_file_index in data.next_train():
      gc.collect()
      logits = model.forward(
        x, x_mask, x_len, x_pos_emb_idxs,
        y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_char, y_char, file_idx=dev_file_index, step=step)
      logits = logits.view(-1, model.hparams.trg_vocab_size)
      labels = y[:,1:].contiguous().view(-1)
      val_loss, val_acc = get_performance(crit, logits, labels, model.hparams, sum_loss=False)
      y_len = torch.FloatTensor(y_len)
      if cuda: y_len = y_len.cuda()
      ave_loss = val_loss.sum(-1) / y_len
      for i in range(model.hparams.batch_size):
        out.write("{}\n".format(ave_loss[i].item()))
      if eop: break
      if step == 10: break
      if step % 1000 == 0: print("{} {}".format(lan, step))
      step += 1

def sim_by_ngram(base_lan, lan_lists):
  base_vocab = "data/{}_eng/ted-train.mtok.{}.ochar4vocab".format(base_lan, base_lan)
  base_vocab_set = set([])
  with open(base_vocab, "r") as myfile:
    for line in myfile:
      base_vocab_set.add(line.strip())

  for lan in lan_lists:
    train = open("data/{}_eng/ted-train.mtok.{}".format(lan, lan), "r")
    out = open("data/{}_eng/ted-train.mtok.{}.sim-ngram".format(lan, lan), "w")
    for line in train:
      words = line.split()
      sim = 0
      for w in words:
       s = 0
       for l in range(1, len(w)):
         for i in range(len(w)-l+1):
           if w[i:i+l] in base_vocab_set: s += 1
       sim += (s / len(w))
      out.write("{}\n".format(sim / len(words)))

def sim_by_ngram_v1(base_lan, lan_lists):
  base_vocab = "data/{}_eng/ted-train.mtok.{}.ochar4vocab".format(base_lan, base_lan)
  base_vocab_set = set([])
  with open(base_vocab, "r") as myfile:
    for line in myfile:
      base_vocab_set.add(line.strip())

  for lan in lan_lists:
    train = open("data/{}_eng/ted-train.mtok.{}".format(lan, lan), "r")
    out = open("data/{}_eng/ted-train.mtok.{}.sim-ngram_v1".format(lan, lan), "w")
    for line in train:
      words = line.split()
      sim = 0
      for w in words:
       s = 0
       for l in range(1, len(w)):
         for i in range(len(w)-l+1):
           if w[i:i+l] in base_vocab_set: s += 1
       sim += s
      out.write("{}\n".format(sim / len(words)))

def sim_gram_all(lan_list_file):
  out = open("ted-train-all.mtok.sim-ngram.graph", "w")
  lan_lists = []
  with open(lan_list_file, 'r') as myfile:
    for line in myfile:
      lan_lists.append(line.strip())
  print("building graph with {} nodes..".format(len(lan_lists)))

  for base_lan in lan_lists:
  #for base_lan in ["bel"]:
    base_vocab = "data/{}_eng/ted-train.mtok.{}.char4vocab".format(base_lan, base_lan)
    if not os.path.isfile(base_vocab):
      print("vocab for {} not exist..".format(base_lan))
      continue
    base_vocab_set = set([])
    with open(base_vocab, "r") as myfile:
      for line in myfile:
        base_vocab_set.add(line.strip())
    print("process base lan {}".format(base_lan))
    for lan in lan_lists:
      train = open("data/{}_eng/ted-train.mtok.{}".format(lan, lan), "r")
      total_sim, count = 0, 0
      for line in train:
        count += 1
        words = line.split()
        sim = 0
        for w in words:
         s = 0
         for l in range(1, len(w)):
           for i in range(len(w)-l+1):
             if w[i:i+l] in base_vocab_set: s += 1
         sim += (s / len(w))
        total_sim += (sim / len(words))
      out.write("{} {} {}\n".format(base_lan, lan, total_sim / count))
      print("process ref lan {}".format(lan))

def sim_vocab_all(lan_list_file):
  #out = open("ted-train-vocab.rtok.sim-ngram.graph", "w")
  out = open("ted-iwslt-vocab.mtok.sim-ngram.graph", "w")
  lan_lists = []
  with open(lan_list_file, 'r') as myfile:
    for line in myfile:
      lan_lists.append(line.strip())
  print("building graph with {} nodes..".format(len(lan_lists)))

  for base_lan in lan_lists:
  #for base_lan in ["bel"]:
    #base_vocab = "data_rtok/{}_eng/ted-train.mtok.{}.char4vocab".format(base_lan, base_lan)
    base_vocab = data_dir + "{}_en/ted-train.rtok.{}.char4vocab".format(base_lan, base_lan)
    if not os.path.isfile(base_vocab):
      print("vocab for {} not exist..".format(base_lan))
      continue
    base_vocab_set = set([])
    with open(base_vocab, "r") as myfile:
      count = 0
      for line in myfile:
        base_vocab_set.add(line.strip())
        count += 1
        if count == 10000: break 

    print("process base lan {}".format(base_lan))
    for lan in lan_lists:
      train_vocab = open(data_dir + "{}_en/ted-train.rtok.{}.char4vocab".format(lan, lan), "r")
      total_sim, count = 0, 0
      for line in train_vocab:
        count += 1
        word = line.strip()
        if word in base_vocab_set: total_sim += 1
        if count == 10000: break
      out.write("{} {} {}\n".format(base_lan, lan, total_sim))
      print("process ref lan {}".format(lan))

def sim_sw_vocab_all(lan_list_file):
  out = open("ted-train-vocab.mtok.sim-spm8000.graph", "w")
  lan_lists = []
  with open(lan_list_file, 'r') as myfile:
    for line in myfile:
      lan_lists.append(line.strip())
  print("building graph with {} nodes..".format(len(lan_lists)))

  for base_lan in lan_lists:
  #for base_lan in ["bel"]:
    base_vocab = "data/{}_eng/ted-train.mtok.spm8000.{}.vocab".format(base_lan, base_lan)
    if not os.path.isfile(base_vocab):
      print("vocab for {} not exist..".format(base_lan))
      continue
    base_vocab_set = set([])
    with open(base_vocab, "r") as myfile:
      count = 0
      for line in myfile:
        base_vocab_set.add(line.strip())
        count += 1
        if count == 1000: break 

    print("process base lan {}".format(base_lan))
    for lan in lan_lists:
      train_vocab = open("data/{}_eng/ted-train.mtok.spm8000.{}.vocab".format(lan, lan), "r")
      total_sim, count = 0, 0
      for line in train_vocab:
        count += 1
        word = line.strip()
        if word in base_vocab_set: total_sim += 1
        if count == 1000: break
      out.write("{} {} {}\n".format(base_lan, lan, total_sim))
      print("process ref lan {}".format(lan))



if __name__ == "__main__":
  #sent_vocab_overlap()
  prob_by_vocab_overlap_sent()
  #prob_by_rank()
  #sim_sw_vocab_all("langs.txt")
  #sim_vocab_all("iwslt_langs.txt")
  #sim_gram_all("langs.txt")
  #prob_by_classify()
  #sim_by_ngram_v1(base_lan, lan_lists)
  #sim_by_ngram(base_lan, lan_lists)
  #sim_by_model("outputs_exp1/semb-8000_azetur_v2/")
