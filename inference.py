from my_gpt import my_gpt
from tokenizer.tokenizer import BPE
import torch

tokenizer = BPE()

device = torch.device("cpu")


model = my_gpt.load_pretrained('model/model_1000_.bin')

params= 0
for p in model.parameters():
    params += p.numel()

print("Parameters {}M".format(params/1e6))



# input_ = "Tak"
input_ = "भाजपा ने दी चेतावनी तो सिद्धारमैया ने BJP नेता का ही मांग लिया इस्तीफा; कर्नाटक में 'MUDA' पर क्यों मचा है बवाल?"
tokens = tokenizer.encode(input_)
print("Sentence {}".format(tokens))
with torch.no_grad():
    gen_ids = model.generate(torch.tensor([tokens], dtype=torch.int32))

print("gen_ids ",gen_ids[0])
output = tokenizer.decode(gen_ids[0].tolist())
print("output", output)