from model import GPT_Model, GPTConfig
import tiktoken
import torch

gpt_config = GPTConfig(
    dropout= 0.2,
    bias= False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model = GPT_Model(gpt_config)
enc = tiktoken.encoding_for_model('gpt-2')
print(gpt_model)
encoded_text = "So this is what it means to.."
output = torch.tensor(enc.encode(encoded_text)).view(1,-1)
print(output.shape)
gpt_model(output)
next_token = gpt_model.generate(output, max_new_tokens=2)
