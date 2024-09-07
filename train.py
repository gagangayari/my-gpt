#%%
from my_gpt import my_gpt
from tokenizer.tokenizer import BPE
import torch
# import argparse
from torch.optim.lr_scheduler import LinearLR
import json
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

# device = torch.device('cpu')


#%%Data
with open('tinyStories.txt', 'r') as file:
    data = file.read()

with open('config.json', 'r') as file:
    config = json.load(file)

block_size = config.get('block_size')

#%% Tokenize
tokenizer = BPE()


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

#%%Estimate loss
eval_iters = 20
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch_data(64, data)
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out
        

          


#%%Training
learning_rate = 3e-4
batch_size = 30
itr = 1000
writer.add_scalar('learning_rate', learning_rate)
writer.add_scalar('batch_size', batch_size)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lambda1 = lambda epoch: 0.85 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)



tokens = tokenizer.encode(data)

# print("token len",len(tokens))

for i in range(itr+1):
    x, y = get_batch_data(batch_size, tokens)
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x,y)
    print("Epoch {} Loss: {}".format(i, loss))
    writer.add_scalar('Loss',loss,i)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if(i%200 == 0):
        model.save_pretrained(f'model/model_{i}_.bin')

#%%Generating

tokens = tokenizer.encode("The news is, sir,")
print("INput", tokens)
gen_ids = model.generate(torch.tensor([tokens], dtype=torch.long).to(device))
print("out", gen_ids)

output = tokenizer.decode(gen_ids[0].tolist())
print("output", output)
# %%


def start_train():
    model = my_gpt()
    model.load_pretrained(args.model_name_or_path)
    model.to(device)
    ## optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    ##train step
    for epoch in range(args.epochs):
        x, y = get_batch_data(64, tokens)
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x,y)
        print("Epoch {} Loss: {}".format(epoch, loss))
        writer.add_scalar('Loss',loss,epoch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()


# if __name__ == "__main__":
#     args = argparse.ArgumentParser()
#     args.add_argument("--model_name_or_path", default='model/model.bin', type=str)
#     args.add_argument("--tokenizer", default='tokenizer/', type=str)
#     args.add_argument("--epochs", default=200, type=int)
#     args.add_argument("--learning_rate", default=3e-4, type=float)
#     args.add_argument("--eval_interval", default=500, type=int)

#     args = args.parse_args()
#     print("Starting heree")
#     start_train()

