import json

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    # s = replace_control_characters(s)
    return s


def get_freq_pairs(inp_toks):
  """Returns a count of the pairs"""
  count = {}
  for pair in zip(inp_toks, inp_toks[1:]):
    count[pair] = count.get(pair,0) + 1
  return count


def merge(id_list, pair, replace_with_idx):
  """
  Replace the occurence of 'pair' in 'id_list' with 'replace_with_idx'

  id_list : List of tokens
  pair : List of 2 numbers
  replace_with_idx : Int value

  Returns new list with the pair replaced
  """
  i=0
  new_ids_list = []
  while(i<len(id_list)):
    if(i<len(id_list)-1 and id_list[i]==pair[0] and id_list[i+1]==pair[1]):
      new_ids_list.append(replace_with_idx)

      i+=2
    else:
      new_ids_list.append(id_list[i])
      i+=1

  return new_ids_list

class Tokenizer():
  def __init__(self):
    self.merges = {}
    ##vocab -> (int) : bytes . For all ints (0-256, 256+ from new merges)

    self.vocab = {}
  
  
  
  def save(self):
    with open('merges.txt', 'w') as f:
      ##Write only the pairs. Not the index of the merged pairs.
      ##When the tokenizer is loaded, allow the user to specify the index
      for p1,p2 in self.merges.keys():
        f.write(f"{p1} {p2}\n")


    with open('vocab.txt', 'w') as f:
      for idx, byte in self.vocab.items():
        s = render_token(byte)
        f.write(f"{idx} {s}\n")

  def load(self):
    try:
      with open('vocab.txt', 'r') as f:
        self.vocab = f.read()

      with open('merges.txt', 'r') as f:
        self.merges = f.read()

    except Exception as e:
      print(e)
      
    


if __name__ == '__main__':
  print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))


