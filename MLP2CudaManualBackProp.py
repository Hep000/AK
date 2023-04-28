# learning from a file of names
# this program will generate more namelike strings
# using a Multilevel Perceptron
# modelled on Bengio et al "A Neural Probalistic Language Model" 2003
# jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

import time
start = time.time()
print('\nImporting libraries\n')
import torch
print('   torch               imported')
import torch.nn.functional as F
print('   torch.nn.functional imported')
# import matplotlib.pyplot as plt
# print('   matplotlib,pyplot   imported')
import random
print('   random              imported')
end = time.time()
print(f'\nElapsed time = {end - start:.3f}\n')

# read list of names
start = time.time()
print('Reading list of names')
words = open('names.txt','r').read().splitlines()
# print('First eight names:\n', words[:8], '\n')
end = time.time()
print(f'Elapsed time = {end - start:.3f}\n')

# mappings of letters in the alphabet (a <-> 1)
start = time.time()
print('Creating character to integer mappings')
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
end = time.time()
print(f'Elapsed time = {end - start:.3f}\n')

# build training, development and test data sets -------------------------------------------------
start = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for block_size in (2, 3, 4): # context length: how many characters do we take to predict the next one?
block_size = 3

print('Building training, development and test data sets with block_size =', block_size,'\n')

def build_dataset(words):

    X, Y = [], []

    count = 0
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append
            count += 1

    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)

    return(X, Y)

words = words[:32000]
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr , Ytr  = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte , Yte  = build_dataset(words[n2:])
print(f'   Size of training    X: {Xtr.shape  } training     Y: {Ytr.shape  } Xtr.is_cuda : {Xtr.is_cuda } Ytr.is_cuda : {Ytr.is_cuda }')
print(f'   Size of development X:  {Xdev.shape} development  Y:  {Ydev.shape} Xdev.is_cuda: {Xdev.is_cuda} Ydev.is_cuda: {Ydev.is_cuda}')
print(f'   Size of test        X:  {Xte.shape } test         Y:  {Yte.shape } Xte.is_cuda : {Xte.is_cuda } Yte.is_cuda : {Yte.is_cuda }')
print(f'\n   First 2 training examples:')
print('       ', words[0])
if block_size == 4: 
    for i in range(5   ): print(f'        {itos[Xtr[i,0].item()]}{itos[Xtr[i,1].item()]}{itos[Xtr[i,2].item()]}{itos[Xtr[i,3].item()]} ---> {itos[Ytr[i].item()]}')
else:
    if block_size == 3:
        for i in range(5   ): print(f'        {itos[Xtr[i,0].item()]}{itos[Xtr[i,1].item()]}{itos[Xtr[i,2].item()]} ---> {itos[Ytr[i].item()]}')
    else:
        if block_size == 2:
            for i in range(5   ): print(f'        {itos[Xtr[i,0].item()]}{itos[Xtr[i,1].item()]} ---> {itos[Ytr[i].item()]}')
print('       ', words[1])
if block_size == 4: 
    for i in range(5,13): print(f'        {itos[Xtr[i,0].item()]}{itos[Xtr[i,1].item()]}{itos[Xtr[i,2].item()]}{itos[Xtr[i,3].item()]} ---> {itos[Ytr[i].item()]}')
else:
    if block_size == 3:
        for i in range(5,13): print(f'        {itos[Xtr[i,0].item()]}{itos[Xtr[i,1].item()]}{itos[Xtr[i,2].item()]} ---> {itos[Ytr[i].item()]}')
    else:
        if block_size == 2:
            for i in range(5,13): print(f'        {itos[Xtr[i,0].item()]}{itos[Xtr[i,1].item()]} ---> {itos[Ytr[i].item()]}')
print()
end = time.time()
print(f'Elapsed time = {end - start:.3f}')

# utility function
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

# meta parameters of MLP
vocab_size = 27
n_embd = 10
n_hidden = 64

# set up MLP
start = time.time()
print(f'\nSet up MLP\n\n\
    block_size       = {block_size}\n\
    n_embd            = {n_embd}\n\
    n_hidden = {n_hidden}\n')
g  = torch.Generator(device=device).manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),          generator=g, device=device, requires_grad=True)
W1 = torch.randn((block_size*n_embd, n_hidden), generator=g, device=device, requires_grad=True) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                      generator=g, device=device, requires_grad=True) * 0.1
W2 = torch.randn((n_hidden, vocab_size),        generator=g, device=device, requires_grad=True) * 0.1
b2 = torch.randn(vocab_size,                    generator=g, device=device, requires_grad=True) * 0.1
bngain = torch.randn((1, n_hidden),             generator=g, device=device, requires_grad=True) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden),             generator=g, device=device, requires_grad=True) * 0.1

parameters = (C, W1, b1, W2, b2, bngain, bnbias)
print('    Number of parameters in MLP=', sum(p.nelement() for p in parameters))
end = time.time()
print(f'Elapsed time = {end - start:.3f}\n')

batch_size = 1024
n = batch_size
print(f'\n    batch_size = {batch_size:.2f}\n')

# constuct minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g, device=device)
Xb, Yb = Xtr[ix], Ytr[ix]

# forward pass
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
# Linear layer
hprebn = embcat @ W1 + b1 
# batch norm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*bndiff2.sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(hpreact)
# Linear layer 2
logits = h @ W2 + b2
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 # if I use (1.0 /  counts_sum) I can't get backprop to be exact ...
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

#PyTorch backward pass
for p in parameters:
    p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
          bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
          embcat, emb]:
    t.retain_grad()
loss.backward()
print(loss)
