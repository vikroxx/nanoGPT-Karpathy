#########################################################
#### OLDER IMPLEMENTATION OF BIGRAM MODEL BY KARPATHY ####
##########################################################


import torch
import torch.nn.functional as F

torch.manual_seed(1337)

from tqdm import tqdm

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
print("length of dataset in characters: ", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

## PARAMETER VALUES
batch_size = 16
block_size = 1024
learning_rate = 1e-3
vocab_size = vocab_size
embedding_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


data = torch.tensor(encode(text), dtype=torch.long).to(device=device)
print(data.shape, data.dtype)
# print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this


# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
test_data = data[n:]


train_data[:block_size+1]
print('Train data 0: ',train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
    
    
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')


for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")


m = BigramLanguageModel(vocab_size).to(device)
out, loss= m(xb, yb)
print(out.shape)
print(loss)


optimizer = torch.optim.Adam(m.parameters() , lr=learning_rate)
for step in tqdm(range(2000)):
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step %100 == 0:
        print('\n',loss.item())
print(loss.item())

idx = torch.zeros((1,1), dtype =torch.long).to(device= device)
print(decode(m.generate(idx, max_new_tokens=2000)[0].tolist()))
