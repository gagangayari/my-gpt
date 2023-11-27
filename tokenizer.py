import torch
import pickle

class my_tokenizer:
    def __init__(self, path, text_corpus_file='input.txt'):
        try:
            print('Initializing tokenizer..')
            self.load(path)
        except Exception as e:
            print(e) 
            print("Building new tokenizer")
            self.stoi = {}
            self.itos = {}
            self.build_tokenizer(text_corpus=text_corpus_file)


    def build_tokenizer(self, text_corpus):
        with open(text_corpus, 'r') as file:
            data = file.read()
            chars = sorted(list(set(data)))

        for i, ch in enumerate(chars):
            self.stoi[ch] = i
            self.itos[i] = ch

    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, token_ids):
        return "".join(self.itos[id] for id in token_ids)
    
    def save(self, path):
        print("saving...")
        with open(path+'stoi.pkl', 'wb') as file1, open(path+'itos.pkl', 'wb') as file2:
            pickle.dump(self.stoi, file1)
            pickle.dump(self.itos, file2)
    
    def load(self, path):
        print("loading...")
        with open(path+'stoi.pkl', 'rb') as file1, open(path+'itos.pkl', 'rb') as file2:
            self.stoi = pickle.load(file1)
            self.itos = pickle.load(file2)






