import argparse
import pickle as pkl

from model import *
from transformer import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from data_utils import DataUtil

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir", type=str)
parser.add_argument("--options", default='char', help="[char|final|char-no-spe]", type=str)
parser.add_argument("--lex_file", type=str)
parser.add_argument("--out_file", type=str)
parser.add_argument("--cuda", action="store_true")

args = parser.parse_args()

model_file_name = os.path.join(args.model_dir, "model.pt")
if not args.cuda:
  model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
else:
  model = torch.load(model_file_name)
model.eval()

src1_w = []
with open(args.lex_file.split(",")[0], "r") as myfile:
  for line in myfile:
    src1_w.append(line.strip())

model.hparams.train_src_file_list = args.lex_file.split(",")
model.hparams.train_trg_file_list = args.lex_file.split(",")
model.hparams.dev_src_file = args.lex_file.split(",")[0]
model.hparams.dev_trg_file = args.lex_file.split(",")[0]
model.hparams.shuffle_train = False
model.hparams.batcher = "sent"
model.hparams.batch_size = len(src1_w) 
model.hparams.cuda = args.cuda

data = DataUtil(model.hparams, shuffle=False)

out = args.out_file.split(",")
out1 = open(out[0], "wb")
out2 = open(out[1], "wb")

step = 0
while True:
  gc.collect()
  x_train, x_mask, x_count, x_len, x_pos_emb_idxs, \
	y_train, y_mask, y_count, y_len, y_pos_emb_idxs, \
	batch_size, x_train_char_sparse, y_train_char_sparse, eop, file_idx = data.next_train()
  if args.options == 'char-no-spe':
    model.hparams.sep_char_proj = False
    print(x_train[44])
    print(x_train_char_sparse[44])
    with torch.no_grad():
      char_emb = model.encoder.char_emb(
              x_train_char_sparse)
  elif args.options == 'char':
    with torch.no_grad():
      char_emb = model.encoder.char_emb(
              x_train_char_sparse, file_idx=file_idx)
  elif args.options == 'final':
    with torch.no_grad():
      char_emb = model.encoder.char_emb(
              x_train_char_sparse, file_idx=file_idx)
      char_emb = model.encoder.word_emb(char_emb, x_train, file_idx=file_idx)
  char_emb = char_emb[:,1,:].squeeze(1).numpy()
  if step == 0:
    pkl.dump(char_emb, out1)
  elif step == 1:
    pkl.dump(char_emb, out2)
  step += 1
  if eop: break

assert step == 2
