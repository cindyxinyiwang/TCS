import re
import sys
import unicodedata


MERGESYMBOL = '↹'


def is_weird(c):
  return not (unicodedata.category(c)[0] in "LMN" or c.isspace())  # Caution: python's isalnum(c) does not accept Marks (category M*)!


def check_for_at(instring):
  for match in re.finditer(' ' + MERGESYMBOL, instring):
    if match.end() < len(instring) and is_weird(instring[match.end()]):
      print("CAUTION: looks like a merge to the detokenizer:", instring[match.start():match.end() + 1], " at position", match.start(), file = sys.stderr)
  for match in re.finditer(MERGESYMBOL + ' ', instring):
    if match.start() > 0 and is_weird(instring[match.start() - 1]):
      print("CAUTION: looks like a merge to the detokenizer:", instring[match.start() - 1:match.end()], " at position", match.start() - 1, file = sys.stderr)


def tokenize(instring):
  # Walk through the string!
  outsequence = []
  for i in range(len(instring)):
    c = instring[i]
    c_p = instring[i - 1] if i > 0 else c
    c_n = instring[i + 1] if i < len(instring) - 1 else c

    # Is it a letter (i.e. Unicode category starts with 'L')?
    # Or alternatively, is it just whitespace?
    # So if it's not weird, just copy.
    if not is_weird(c):
      outsequence.append(c)
    # Otherwise it should be separated!
    else:
      # Was there something non-spacey before?
      # Then we have to introduce a new space and a merge marker.
      if not c_p.isspace():
        outsequence.append(' ' + MERGESYMBOL)
      # Copy character itself
      outsequence.append(c)
      # Is there something non-spacey after?
      # Then we have to introduce a new space and a merge marker.
      # If, however the next character would just want to merge left anyway, no need to do it now.
      if not c_n.isspace() and not is_weird(c_n):
        outsequence.append(MERGESYMBOL + ' ')

  return ''.join(outsequence)


def detokenize(instring):
  # Walk through the string!
  outsequence = []
  i = 0
  while i < len(instring):
    c = instring[i]
    c_n = instring[i + 1] if i < len(instring) - 1 else c
    c_nn = instring[i + 2] if i < len(instring) - 2 else c

    # It could be one of the spaces we introduced
    if c + c_n == ' ' + MERGESYMBOL and is_weird(c_nn):
      i += 2
    elif is_weird(c) and c_n + c_nn == MERGESYMBOL + ' ':
      outsequence.append(c)
      i += 3
    else:
      outsequence.append(c)
      i += 1

  return ''.join(outsequence)


if sys.argv[1] == '--tok':
  instring = sys.stdin.read()
  check_for_at(instring)
  tok_string = tokenize(instring)
  if detokenize(tok_string) != instring:
    print("Incorrectness somewhere :(", file = sys.stderr)
  sys.stdout.write(tok_string)
elif sys.argv[1] == '--detok':
  instring = sys.stdin.read()
  sys.stdout.write(detokenize(instring))
else:
  print('First parameter has to be --tok or --detok!', file = sys.stderr)
  exit(1)

