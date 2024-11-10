from my_gpt import my_gpt
from tokenizer.tokenizer import BPE
from transformers import AutoTokenizer
import torch

tokenizer = BPE()
# tokenizer = AutoTokenizer.from_pretrained('../../phi-2')
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"
device = torch.device("cpu")


model = my_gpt.load_pretrained('model_new_arch/model_1000_.bin')

params= 0
for p in model.parameters():
    params += p.numel()

print("Parameters {}M".format(params/1e6))



# input_ = "भाजपा ने दी चेतावनी तो सिद्धारमैया ने BJP नेता का ही मांग लिया इस्तीफा; कर्नाटक में 'MUDA' पर क्यों मचा है बवाल?"
input_ = "He "
tokens = tokenizer.encode(input_)
# tokens = tokenizer(input_, return_tensors="pt", padding="max_length",truncation=True, max_length=128)['input_ids']
print(f"Length of tokens {len(tokens)}" )
# print("Sentence {}".format(tokens))

with torch.no_grad():
    gen_ids = model.generate(torch.tensor([tokens], dtype=torch.int32))
    # gen_ids = model.generate(tokens)


output = tokenizer.decode(gen_ids[0].tolist())
print("output", output)