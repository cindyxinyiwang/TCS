## filter out repeated sentences
lan = "ces"
src = "data/{}_eng/ted-train.mtok.{}".format(lan, lan)
src_out = "data/{}_eng/ted-train.mtok.{}.filt_rep".format(lan, lan)
trg = "data/{}_eng/ted-train.mtok.spm8000.eng".format(lan)
trg_out = "data/{}_eng/ted-train.mtok.spm8000.eng.filt_rep".format(lan)

srcs = set()
trgs = set()
src_out = open(src_out, "w")
trg_out = open(trg_out, "w")
src = open(src, "r")
trg = open(trg, "r")
total, filt = 0, 0
for s, t in zip(src, trg):
  total += 1
  if not t in trgs and not s in srcs:
    trgs.add(t)
    srcs.add(s)
    src_out.write(s)
    trg_out.write(t)
  else:
    filt += 1
print("total: {}, filtered: {}".format(total, filt))
