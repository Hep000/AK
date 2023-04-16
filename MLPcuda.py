# learning from a file of names
# this program will generate more namelike strings
# using a Multilevel Perceptron
# modelled on Bengio et al "A Neural Probalistic Language Model" 2003
# jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

import torch
import torch.nn.functional as F

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
count = 0
X, Y = [], []
for w in words:
    
    count += 1
    if count <=8: print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        if count <=8: print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)

g = torch.Generator(device=device).manual_seed(2147483647) # for reproducibility
C = torch.randn((27,2),    generator=g, device=device, requires_grad=True)
W1 = torch.randn((6,100),  generator=g, device=device, requires_grad=True)
b1 = torch.randn(100,      generator=g, device=device, requires_grad=True)
W2 = torch.randn((100,27), generator=g, device=device, requires_grad=True)
b2 = torch.randn(27,       generator=g, device=device, requires_grad=True)

parameters = (C, W1, b1, W2, b2)
print('Number of parameters =', sum(p.nelement() for p in parameters))

for _ in range(100):

    # forward pass
    emb = C[X]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    print('loss =', round(loss.item(),2))

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    for p in parameters:
        p.data += -0.1 * p.grad

