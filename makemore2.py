words = open('names.txt', 'r').read().splitlines()
import torch
# a = torch.zeros((3,5))
# a[1,3] = 1
# print(a)
# print(a.dtype)
# a[1,3] += 1
# print(a)
# a[0,0] = 5
# print(a)
N = torch.zeros((28,28),dtype=torch.int32)
# print(set(''.join(words)))
# print(len(set(''.join(words))))
# print(sorted(list(set(''.join(words)))))
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27
# print(type(stoi),'/n',stoi)
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

import matplotlib.pyplot as plt
# plt.imshow(N)
# plt.show()
itos = {i:s for s,i in stoi.items()}
# print(itos)
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center",va="bottom", color='gray', fontsize=7)
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='grey', fontsize=6)
plt.axis('off')
plt.show()

