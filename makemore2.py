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
N = torch.zeros((27,27),dtype=torch.int32)
# print(set(''.join(words)))
# print(len(set(''.join(words))))
# print(sorted(list(set(''.join(words)))))
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
# print(type(stoi),'/n',stoi)
itos = {i:s for s,i in stoi.items()}
b = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# import matplotlib.pyplot as plt
# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center",va="bottom", color='gray', fontsize=12)
#         plt.text(j, i, N[i, j].item(), ha="center", va="top", color='grey', fontsize=10)
# plt.axis('off')
# plt.show()
# print(N[0])
p = N[0].float()
p = p / p.sum()
# print(p.sum())

g = torch.Generator().manual_seed(2147483647)
# p = torch.rand(3, generator=g)
# p = p / p.sum()
# print(p)
# print(torch.multinomial(p, num_samples=20, replacement=True, generator=g))
ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
print(itos[ix])