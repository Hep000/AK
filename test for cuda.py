# learning form a file of names
# this program will generate more 'names'
# using a Multilevel Perceptron
# modelled on Bengio et al "A Neural Probalistic Language Model" 2003
# jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

import torch
import torch.nn.functional as F
# import matplotlib as plt

print('\n','torch.cuda.is_available =', torch.cuda.is_available(),'\n')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device =', device, '\n')


words = open('names.txt','r').read().splitlines()
print('First eight names:', words[:8], '\n')

# mappings of letters in the alphabet (a <-> 1)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print('Mapping between intergers and letters:', itos, '\n')

# build the dataset

block_size = 3 # context length: how many characters do we take to predict the next one?
print('Examples of the 3 character context and the corresponding label')
X, Y = [], []
for w in words:
    
    # print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

print('X shape and type:', X.shape, X.dtype)
print('Y shape and type:', Y.shape, Y.dtype)

# randomly embed X into a 2D space
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
emb = C[X].to(device)
print('emb.is_cuda =', emb.is_cuda, '\n')
print('emb shape:', emb.shape)

# hidden layer
W1 = torch.randn((6,100), generator=g).to(device)
b1 = torch.randn(100, generator=g).to(device)
print('W1.is_cuda =', W1.is_cuda)
print('b1.is_cuda =', b1.is_cuda)

# output from hidden layer
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
print('h.is_cuda =', h.is_cuda)
print(h[0,:])

# output layer
W2 = torch.randn((100, 27), generator=g).to(device)
b2 = torch.randn(27, generator=g).to(device)
print('W2.is_cuda =', W2.is_cuda)
print('b2.is_cuda =', b2.is_cuda)

# output
logits = h @ W2 + b2
print('logits.is_cuda =', logits.is_cuda)

