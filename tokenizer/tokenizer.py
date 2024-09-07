from .base import get_freq_pairs, merge, Tokenizer

class BPE(Tokenizer):
    def __init__(self) -> None:
        super().__init__()
    
    def train(self, vocab_size, text):
        ##Vocabulary should contain atleast the ASCII characters
        assert vocab_size>=256

        num_merges = vocab_size-256
        tokens = list(text.encode('utf-8'))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = get_freq_pairs(tokens)
            max_pair = max(stats, key=stats.get)
            idx = 256 + i
            tokens = merge(tokens, max_pair, idx)
            merges[max_pair] = idx 
            vocab[idx] = vocab[max_pair[0]] + vocab[max_pair[1]]

            
        self.merges = merges
        self.vocab = vocab

        self.save()

    def encode(self, text):
        ids = list(text.encode('utf-8'))    
        # print(ids)
        # assert len(self.merges) > 0
        ##if len(ids) is greater than 2, we need to merge it
        while True:
            pair_counts = get_freq_pairs(ids)
            # print(pair_counts)
        
            min_index_pair = min(pair_counts, key= lambda x: self.merges.get(x, float('inf')))
            if(min_index_pair) not in self.merges:
                break

            idx = self.merges.get(min_index_pair)
            # print(ids)
            ids = merge(ids, min_index_pair, idx)
        return ids

    def decode(self, ids):
        print(ids)
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


if __name__ == "__main__":

    tokenizer = tokenizer()

    with open('cindrella_stories.txt', 'r') as f:
        text = f.read()


    tokenizer.train(500, text)

    s = "😁"
    print("String is",s)

    ids  = tokenizer.encode(s)
    print("Encoded string ",ids)
    decoded_string = tokenizer.decode(ids)
    print("Decoded string ",decoded_string)