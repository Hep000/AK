words = open('names.txt', 'r').read().splitlines()
import torch
N = torch.zeros((27,27),dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
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

P = (N+1).float()
P /= P.sum(1, keepdim=True)
print(P.shape)
print(P.sum(1, keepdim=True).shape)

g = torch.Generator().manual_seed(2147483647)

for i in range(5):

    out = []
    ix = 0
    while True:
        # p = N[ix].float()
        # p = p / p.sum()
        p = P[ix,:]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

loglikelyhood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        loglikelyhood += logprob
        n += 1
        # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
print(f'{loglikelyhood=}')
nll = - loglikelyhood
print(f'{nll=}')
print(f'{nll/n=}')