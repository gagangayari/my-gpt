#%%
from my_gpt import my_gpt
from tokenizer import my_tokenizer
import torch
from torch.optim.lr_scheduler import LinearLR
import json

device = torch.device('cpu')

#%%Data
with open('input.txt', 'r') as file:
    data = file.read()

with open('config.json', 'r') as file:
    config = json.load(file)

block_size = config.get('block_size')

#%% Tokenize
tokenizer = my_tokenizer('tokenizer/')

def get_batch_data(batch_size, data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    return x, y


#%%Model

model = my_gpt()


model.to(device)

params= 0
for p in model.parameters():
    params += p.numel()

print("Parameters {}M".format(params/1e6))

    


#%%Training
learning_rate = 3e-6
epochs = 20

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lambda1 = lambda epoch: 0.85 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)



tokens = tokenizer.encode(data)

print("token len",len(tokens))

for i in range(epochs):
    x, y = get_batch_data(20, tokens)
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x,y)
    print("Epoch {} Loss: {}".format(i, loss))
    loss.backward()
    optimizer.step()
    print("learning rate : ",scheduler.get_lr())
    scheduler.step()

model.save_pretrained('model/model.bin')

#%%Generating

tokens = tokenizer.encode("The news is, sir,")
print("INput", tokens)
gen_ids = model.generate(torch.tensor([tokens], dtype=torch.int32))
print("out", gen_ids)

output = tokenizer.decode(gen_ids[0].tolist())
print("output", output)
# %%
