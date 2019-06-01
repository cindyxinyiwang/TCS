import sys
import io

#input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
#input_stream = open(sys.argv[1], encoding="utf-8")

vocab = {}
for line in sys.stdin:
  toks = line.split()
  for t in toks:
    if t not in vocab:
      vocab[t] = 0
    vocab[t] += 1

vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)
print("<pad>")
print("<unk>")
print("<s>")
print("<\s>")

for w, c in vocab:
  print(w)
