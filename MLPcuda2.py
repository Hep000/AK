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
for i in range(7,15): print(f'        {itos[Xtr[i,0].item()]}{itos[Xtr[i,1].item()]}{itos[Xtr[i,2].item()]} ---> {itos[Ytr[i].item()]}')
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

lr_order = torch.zeros((18,3), dtype=int)
lr_order[0 ,0], lr_order[ 0,1], lr_order[ 0,2] = 1, 0, 0 # 1
lr_order[1 ,0], lr_order[ 1,1], lr_order[ 1,2] = 0, 1, 0 # 2
lr_order[2 ,0], lr_order[ 2,1], lr_order[ 2,2] = 0, 0, 1 # 3

lr_order[3 ,0], lr_order[ 3,1], lr_order[ 3,2] = 1, 0, 0 # 1
lr_order[4 ,0], lr_order[ 4,1], lr_order[ 4,2] = 0, 0, 1 # 3
lr_order[5 ,0], lr_order[ 5,1], lr_order[ 5,2] = 0, 1, 0 # 2

lr_order[6 ,0], lr_order[ 6,1], lr_order[ 6,2] = 0, 1, 0 # 2
lr_order[7 ,0], lr_order[ 7,1], lr_order[ 7,2] = 1, 0, 0 # 1
lr_order[8 ,0], lr_order[ 8,1], lr_order[ 8,2] = 0, 0, 1 # 3

lr_order[9 ,0], lr_order[ 9,1], lr_order[ 9,2] = 0, 1, 0 # 2
lr_order[10,0], lr_order[10,1], lr_order[10,2] = 0, 0, 1 # 3
lr_order[11,0], lr_order[11,1], lr_order[11,2] = 1, 0, 0 # 1

lr_order[12,0], lr_order[12,1], lr_order[12,2] = 0, 0, 1 # 3
lr_order[13,0], lr_order[13,1], lr_order[13,2] = 1, 0, 0 # 1
lr_order[14,0], lr_order[14,1], lr_order[14,2] = 0, 1, 0 # 2

lr_order[15,0], lr_order[15,1], lr_order[15,2] = 0, 0, 1 # 3
lr_order[16,0], lr_order[16,1], lr_order[16,2] = 0, 1, 0 # 2
lr_order[17,0], lr_order[17,1], lr_order[17,2] = 1, 0, 0 # 1
# lr_order.to(device)

# optimize over a different optimization parameters
for minibatch_size in (128, 256):

    print('Beginning optimisation with meta parameters:\n')
    print(f'\
    n_hidden_neurons = {n_hidden_neurons  }\n\
    minibatch_size   = {minibatch_size:.2f}')
    print()

    start = time.time()

    # test and update learning rate
    lr_medium = 0.12
    lr_high   = 0.13
    lr_low    = 0.11
    lr_result = torch.zeros((18,3), device=device)
    # lr_result = torch.zeros((18,3))

    while time.time() - start < 20:

        # calculate loss of initial model
        # emb = C[Xtr]
        # h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
        # logits = h @ W2 + b2
        # tr_loss = F.cross_entropy(logits, Ytr)
        tr_loss = 1

        for lr_test_i in range(18):
            loss_start = tr_loss
            if lr_order[lr_test_i, 0] == 1:
                lr = lr_medium
                lr_test_j = 0
            else:
                if lr_order[lr_test_i, 1] == 1:
                    lr = lr_high
                    lr_test_j = 1
                else:
                    if lr_order[lr_test_i, 2] == 1:
                        lr = lr_low
                        lr_test_j = 2
                    else: print('lr_test_error')
                
            for i1000 in range(100):

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

                # update MLP parameters
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

            # store change in loss
            change = (tr_loss - loss_start) / loss_start
            lr_result[lr_test_i, lr_test_j] = change
            print(f'lr_result.is_cuda = {lr_result.is_cuda} change.is_cuda = {change.is_cuda} change = {change:.3f} lr_test_i = {lr_test_i} lr_test_j = {lr_test_j} lr_result[lr_test_i, lr_test_j] = {lr_result[lr_test_i, lr_test_j]}')
            # lr_result[lr_test_i, lr_test_j] = change # store relative change in loss

            # forward pass over development data data
            emb = C[Xdev]
            h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
            logits = h @ W2 + b2
            dev_loss = F.cross_entropy(logits, Ydev)

            print('        iterations so far =', i1000+1, 'lr =', lr,\
                'Loss over training data =', round(tr_loss.item(),3),\
                'Loss over development data =', round(dev_loss.item(),3))#,\
                # 'Relative loss =', round(lr_result[lr_test_i, lr_test_j].item(),3))

        end = time.time()
        print(f'\nElapsed time = {end - start:.1f}\n')
