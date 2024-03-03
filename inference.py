from my_gpt import my_gpt
from tokenizer import my_tokenizer
import torch

tokenizer = my_tokenizer('model/')

model = my_gpt.load_pretrained('model/model.bin')

params= 0
for p in model.parameters():
    params += p.numel()

print("Parameters {}M".format(params/1e6))



input_ = input("Enter a sentence: ")
tokens = tokenizer.encode(input_)
with torch.no_grad():
    gen_ids = model.generate(torch.tensor([tokens], dtype=torch.int32))

output = tokenizer.decode(gen_ids[0].tolist())
print("output", output)