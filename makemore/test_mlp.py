# %% Cell 1
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# %% Cell 2
words = open("names.txt", "r").read().split()
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
print(itos)

# %% Cell 3
block_size = 3
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
X = torch.tensor(X)
Y = torch.tensor(Y)
print(X.shape)
print(Y.shape)
import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = X[:n1], Y[:n1]
Xdev, Ydev = X[n1:n2], Y[n1:n2]
Xte, Yte = X[n2:], Y[n2:]
print(Xtr.shape)
print(Ytr.shape)


# %% Cell 4
"""
NOTE:
    We can see the embedding of the integer as indexing
    into a look up table. But equivalently it is a layer, of weight
    matrix C and it has no no-linearity, it is just neurons
    This was presented by the idea of one_hot then mat mul, but indexing is faster
"""
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((27, 10), generator=g)

# %% Cell 5

W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# %% Cell 6
# Choosing a good lr -> use expenential for a more logical spread of lr
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

# %% Cell 7
lri = []
lossi = []
stepi = []

# %% Cell 8

for i in range(10000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forward
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdim=True)
    # loss = -prob[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[ix])
    # backward
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    # lr = lrs[i]
    lr = 0.1

    for p in parameters:
        p.data += -lr * p.grad

    # # track stats
    # lri.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item())

# print(loss.item())
# %% Cell 9
# We want to chose a lr that is in the valley
# That way we have good learning and stability, here it happens to coincide with 0.1
# plt.plot(lri, lossi)
plt.plot(stepi, lossi)

# %% Cell 10

emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print(loss.item())
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss.item())


# %% Cell 11
# plt.figure(figsize=(8, 8))
# plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
# for i in range(C.shape[0]):
#     plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color="white")
# plt.grid('minor')
