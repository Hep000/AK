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

# # build training, development and test data sets -------------------------------------------------
start = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

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

words = words[:1000]
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

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g, device=device) / fan_in**0.5
        self.bias = torch.zeros(fan_out, device=device) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)

    def __call__(self, x):
        # calculate the foreward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)     # batch mean
            xvar = x.var(0, keepdim=True, unbiased=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var - (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out                    
                 
    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

vocab_size = 27    
n_embd = 10 # the dimensiality of the character embedding vectors
n_hidden = 100 # number of neurons in the hidden layer
g = torch.Generator(device=device).manual_seed(2147483647) # for reproducibility

C = torch.randn((vocab_size, n_embd), generator=g, device=device)
layers = [
    Linear(n_embd * block_size, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, vocab_size), BatchNorm1d(vocab_size),
]

with torch.no_grad():
    # last layer: make less confident
    layers[-1].gamma *= 0.1
    # all other layers apply gain
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
n_parameters = sum(p.nelement() for p in parameters)
print(n_parameters) # number of parameters in total
for p in parameters:
    p.requires_grad = True

max_steps = 1000
batch_size = 1024
lossi = []
ud = []

for i_step in range(max_steps):

    # if i_step == 0:
    #     batch_size = 32 # to closer match AK's gradient distribution figures and graph
    # else:
    #     batch_size = 1024 # for faster convergence using the GPU
    
    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g, device=device)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    for layer in layers:
        layer.out.retain_grad() # AFTER_DEBUG: would take out retain_grad
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 4000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad
    
    # track stats
    if i % 1000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    p_list = [] # list for parameter p
    ud_folded = []
    with torch.no_grad():
        for p in parameters:
            t_grad_cuda = p.grad; t_grad = t_grad_cuda.to(device_cpu)
            t_data_cuda = p.data; t_data = t_data_cuda.to(device_cpu)
            ud_folded.append((lr*t_grad.std() / t_data.std()).log10().item())
            # ud.append([(lr*t.grad.std() / t.data.std()).log10().item() for p in parameters])
        ud.append(ud_folded)

    if i_step == 0:
        # visualize histograms - activation distribution
        plt.figure(figsize=(20, 4)) # width and height of the plot
        legends = []
        for i, layer in enumerate(layers[:-1]): # note exclude the output layer
            if isinstance(layer, Tanh):
                t_cuda = layer.out
                t = t_cuda.to(device_cpu)
                print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' %\
                    (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'layer {i} ({layer.__class__.__name__})')
        plt.legend(legends)
        plt.title('activation distribution')
        plt.show()

        # visualize histograms - gradient distribution
        plt.figure(figsize=(20, 4)) # width and height of the plot
        legends = []
        for i, layer in enumerate(layers[:-1]): # note exclude the output layer
            if isinstance(layer, Tanh):
                t_cuda = layer.out.grad
                t = t_cuda.to(device_cpu)
                print('layer %d (%10s): mean %+f, std %e' %\
                    (i, layer.__class__.__name__, t.mean(), t.std()))
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'layer {i} ({layer.__class__.__name__})')
        plt.legend(legends)
        plt.title('gradient distribution')
        plt.show()

        # visualize histograms - weights gradient distribution
        plt.figure(figsize=(20, 4)) # width and height of the plot
        legends = []
        for i, p in enumerate(parameters):
            t_cuda = p.grad
            t = t_cuda.to(device_cpu)
            if p.ndim == 2:
                print('weight %10s | mean %+f | std %e | grad:data ration %e' %\
                    (tuple(p.shape), t.mean(), t.std(), t.std() / t.std()))
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'{i} {tuple(p.shape)}')
        plt.legend(legends)
        plt.title('weights gradient distribution')
        plt.show()


    if i_step == 999:
        # visualize plot of update ratios
        plt.figure(figsize=(20, 4)) # width and height of the plot
        legends = []
        for i, p in enumerate(parameters):
            if p.ndim == 2:
                plt.plot([ud[j][i] for j in range(len(ud))])
                legends.append('param %d' % i)
        plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios shoud be ~1e-3, indicate on plot
        plt.legend(legends)
        plt.title('update ratios')
        plt.show()

