from my_gpt import my_gpt
from tokenizer import my_tokenizer
import torch

tokenizer = my_tokenizer('model/')

model = my_gpt.load_pretrained('model/model.bin')

params= 0
for p in model.parameters():
    params += p.numel()

print("Parameters {}M".format(params/1e6))




tokens = tokenizer.encode("The ")
print("INput", tokens)
gen_ids = model.generate(torch.tensor([tokens], dtype=torch.int32))
print("out", gen_ids)

output = tokenizer.decode(gen_ids[0].tolist())
print("output", output)