import argparse
import pickle as pkl
import numpy as np
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
base_lan = "slk"
#lan_lists = ["ukr", "rus", "bul", "mkd", "kaz", "mon"]
#lan_lists = ["aze", "tur", "rus", "por", "ces"]
#lan_lists = ["tur", "ind", "msa", "epo", "sqi", "swe", "dan"]
#lan_lists = ["por", "spa", "ita", "fra", "ron", "epo"]
#lan_lists = ["ces", "slv", "hrv", "bos", "srp"]
cuda = True

data_dir = "/projects/tir3/users/xinyiw1/data_iwslt/" 

def prob_by_lm_sent():
  base_lans = ["aze", "bel", "glg", "slk"]
  ts = [0.01, 0.05, 0.1, 0.1]
  argmaxs = [False, False, False, True]
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
        lm_file = "lmll/ted-train.mtok.{}.{}-lmll".format(lan, base_lan)
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
          log_score = []
          for s in src_list:
            #s[2] = np.exp(-s[2] / t)
            #sum_score += s[2]
            s[2] = -s[2] / t
            log_score.append(s[2])
          sum_score = logsumexp(log_score)
          for s in src_list:
            #s[2] = s[2] / sum_score
            s[2] = np.exp(s[2] - sum_score)
            out_probs[s[0]][s[1]] = s[2]

      for i, lan in enumerate(lan_lists):
        if argmax:
          out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-sent-{}-am".format(lan, lan, base_lan), "w")
        else:
          out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-sent-{}-t{}".format(lan, lan, base_lan, t), "w")
        #out = open(data_dir + "{}_en/ted-train.mtok.{}.prob-rank-{}-t{}-k{}-el".format(lan, lan, base_lan, t, k), "w")
        for p in out_probs[i]:
          out.write("{}\n".format(p))
        out.close()
      if argmax:
        out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-sent-{}-am".format(base_lan, base_lan, base_lan), "w")
      else:
        out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-sent-{}-t{}".format(base_lan, base_lan, base_lan, t), "w")
      #out = open(data_dir + "{}_en/ted-train.mtok.{}.prob-rank-{}-t{}-k{}".format(base_lan, base_lan, base_lan, t, k), "w")
      base_lines = len(open("data/{}_eng/ted-train.mtok.spm8000.eng".format(base_lan)).readlines())
      #base_lines = len(open(data_dir + "{}_en/ted-train.mtok.spm8000.en".format(base_lan)).readlines())
      for i in range(base_lines):
        out.write("{}\n".format(1))
      out.close()

def prob_by_lm_doc():
  base_lans = ["aze", "bel", "glg", "slk"]
  ts = [0.01, 0.05, 0.1, 0.1]
  argmaxs = [False, False, False, True]
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
        lm_file = "lmll/ted-train.mtok.{}.lmll-doc".format(base_lan)
        lm_score = {}
        with open(lm_file, 'r') as myfile:
          for line in myfile:
            toks = line.split()
            lm_score[toks[0]] = float(toks[1])

        trg_file = "data/{}_eng/ted-train.mtok.spm8000.eng".format(lan)
        trg_sents = open(trg_file, 'r').readlines()
        out_probs.append([0 for _ in range(len(trg_sents))])
        line = 0
        for j, trg in enumerate(trg_sents):
          if trg not in trg2srcs: trg2srcs[trg] = []
          trg2srcs[trg].append([i, line, lm_score[lan]])
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
          log_score = []
          for s in src_list:
            #s[2] = np.exp(-s[2] / t)
            #sum_score += s[2]
            s[2] = -s[2] / t
            log_score.append(s[2])
          sum_score = logsumexp(log_score)
          for s in src_list:
            #s[2] = s[2] / sum_score
            s[2] = np.exp(s[2] - sum_score)
            out_probs[s[0]][s[1]] = s[2]

      for i, lan in enumerate(lan_lists):
        if argmax:
          out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-doc-{}-am".format(lan, lan, base_lan), "w")
        else:
          out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-doc-{}-t{}".format(lan, lan, base_lan, t), "w")
        #out = open(data_dir + "{}_en/ted-train.mtok.{}.prob-rank-{}-t{}-k{}-el".format(lan, lan, base_lan, t, k), "w")
        for p in out_probs[i]:
          out.write("{}\n".format(p))
        out.close()
      if argmax:
        out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-doc-{}-am".format(base_lan, base_lan, base_lan), "w")
      else:
        out = open("data/{}_eng/ted-train.mtok.{}.prob-lm-doc-{}-t{}".format(base_lan, base_lan, base_lan, t), "w")
      #out = open(data_dir + "{}_en/ted-train.mtok.{}.prob-rank-{}-t{}-k{}".format(base_lan, base_lan, base_lan, t, k), "w")
      base_lines = len(open("data/{}_eng/ted-train.mtok.spm8000.eng".format(base_lan)).readlines())
      #base_lines = len(open(data_dir + "{}_en/ted-train.mtok.spm8000.en".format(base_lan)).readlines())
      for i in range(base_lines):
        out.write("{}\n".format(1))
      out.close()


if __name__ == "__main__":
  prob_by_lm_sent()
  prob_by_lm_doc()
  #prob_by_rank()
  #sim_sw_vocab_all("langs.txt")
  #sim_vocab_all("iwslt_langs.txt")
  #sim_gram_all("langs.txt")
  #prob_by_classify()
  #sim_by_ngram_v1(base_lan, lan_lists)
  #sim_by_ngram(base_lan, lan_lists)
  #sim_by_model("outputs_exp1/semb-8000_azetur_v2/")
