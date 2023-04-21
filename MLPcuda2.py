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
import matplotlib.pyplot as plt
print('   matplotlib,pyplot   imported')
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

block_size = 3 # context length: how many characters do we take to predict the next one?

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
print('        ', words[0])
for i in range(7): print(f'        {itos[Xtr[i,0].item()]}{itos[Xtr[i,1].item()]}{itos[Xtr[i,2].item()]} ---> {itos[Ytr[i].item()]}')
print('        ', words[1])
for i in range(7,1,15): print(f'        {itos[Xtr[i,0].item()]}{itos[Xtr[i,1].item()]}{itos[Xtr[i,2].item()]} ---> {itos[Ytr[i].item()]}')
print()
end = time.time()
print(f'Elapsed time = {end - start:.3f}\n')

# meta parameters of MLP
dim_C = 2
n_hidden_neurons = 100

# set up MLP
start = time.time()
print('Set up MLP\n')
g = torch.Generator(device=device).manual_seed(2147483647) # for reproducibility
C = torch.randn((27,dim_C),    generator=g, device=device, requires_grad=True)
W1 = torch.randn((block_size*dim_C,n_hidden_neurons),  generator=g, device=device, requires_grad=True)
b1 = torch.randn(n_hidden_neurons,      generator=g, device=device, requires_grad=True)
W2 = torch.randn((n_hidden_neurons,27), generator=g, device=device, requires_grad=True)
b2 = torch.randn(27,       generator=g, device=device, requires_grad=True)

parameters = (C, W1, b1, W2, b2)
print('   Number of parameters in MLP=', sum(p.nelement() for p in parameters),'\n')
end = time.time()
print(f'Elapsed time = {end - start:.3f}\n')


# optimize over a different meta parameters
for minibatch_size in (32, 64, 128):

    print('Beginning optimisation with meta parameters:\n')
    lr_first__10000 = 0.1
    lr_second_10000 = 0.05
    lr_third__10000 = 0.02
    lr_final_10000s = 0.01
    print(f'\
    n_hidden_neurons  = {n_hidden_neurons   }\n\
    minibatch_size    = {minibatch_size     }\n\
    lr_first__10,000  = {lr_first__10000:.2f}\n\
    lr_second_10,000  = {lr_second_10000:.2f}\n\
    lr_third__10,000  = {lr_third__10000:.2f}\n\
    lr_final_10,000s  = {lr_final_10000s:.2f}')
    print()

    start = time.time()
    i10000 = 0
    while i10000 <=5 and time.time() - start < 20:

        for _ in range(1000):

            # constuct minibatch
            ix = torch.randint(0, Xtr.shape[0], (minibatch_size,), generator=g, device=device)

            # forward pass
            emb = C[Xtr[ix]]
            h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, Ytr[ix])

            # backward pass
            for p in parameters:
                p.grad = None
            loss.backward()

            # update
            if i10000 == 0: 
                lr = lr_first__10000
            else    :
                if i10000 == 1: 
                    lr = lr_second_10000
                else: 
                    if i10000 == 2:
                        lr = lr_third__10000
                    else:
                        lr = lr_final_10000s
            for p in parameters:
                p.data += -lr * p.grad

            # track statistics
            # lri.append(lr)
            # lossi.append(loss)

        # forward pass over training data data
        emb = C[Xtr]
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
        logits = h @ W2 + b2
        tr_loss = F.cross_entropy(logits, Ytr)

        # forward pass over development data data
        emb = C[Xdev]
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
        logits = h @ W2 + b2
        dev_loss = F.cross_entropy(logits, Ydev)

        print('        iterations so far =', (i10000+1)*10000, 'Last 10000 useing lr =', lr,\
            'Loss over training data =', round(tr_loss.item(),3),\
            'Loss over development data =', round(dev_loss.item(),3))
        
        i10000 += 1


    end = time.time()
    print(f'\nElapsed time = {end - start:.1f}\n')
