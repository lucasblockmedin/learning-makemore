# backprop ninja

# %% Cell 1
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# %% Cell 2
words = open("names.txt", "r").read().splitlines()
print(len(words))


# %% Cell 3

chars = sorted(list(set("".join(words))))
print(chars)
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
print(stoi)
print(itos)
vocab_size = len(itos)
print(vocab_size)
# %% Cell 4

block_size = 3


def build_dataset(words):
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
    return X, Y


import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


# %% Cell 5
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(
        f"{s:15s} | exact: {str(ex):5s} | approx: {str(app):5s} | max diff: {maxdiff:.4f}"
    )


# %% Cell 6

n_embd = 18  # dimensionality of char embedding vectors
n_hidden = 64  # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)

W1 = (
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / ((n_embd * block_size) ** 0.5)
)
b1 = (
    torch.randn(n_hidden, generator=g) * 0.1
)  # becquse of batch norm this is actually useless

W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1

bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

"""
Note:
    vars started to 0 could mask a incorrect implementation of a gradient
    so we set to just small values
"""

# %% Cell 7

batch_size = 32
n = batch_size
# construct a mini batch
ix = torch.randint(0, Xtr.shape[0], (n,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# %% Cell 8

emb = C[Xb]  # embed chars into vectors
embcat = emb.view(emb.shape[0], -1)  # concat the vectors
hprebn = embcat @ W1 + b1
bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff.pow(2)
bnvar = 1 / (n - 1) * bndiff2.sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
h = torch.tanh(hpreact)
logits = h @ W2 + b2

logit_maxes = logits.max(1, keepdim=True).values
norm_logits = (
    logits - logit_maxes
)  # numerical stability trick, no explosion of exponential
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# Pytorch backwards
for p in parameters:
    p.grad = None
for t in [
    logprobs,
    probs,
    counts,
    counts_sum,
    counts_sum_inv,
    norm_logits,
    logit_maxes,
    logits,
    h,
    hpreact,
    bnraw,
    bndiff,
    bndiff2,
    bnvar,
    bnvar_inv,
    bnmeani,
    hprebn,
    embcat,
    emb,
]:
    t.retain_grad()

loss.backward()
loss

# %% Cell 8

# Exercise 1 backprop through the whole thing manually

dlogprobs = torch.zeros((32, 27))
dlogprobs[range(n), Yb] = -1.0 / n
cmp("logprobs", dlogprobs, logprobs)

dprobs = (1.0 / probs) * dlogprobs
cmp("probs", dprobs, probs)
dcounts_sum_inv = (counts * dprobs).sum(
    1, keepdim=True
)  # because trhe operation is actually broadcasting, then multiplying
cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)

dcounts_sum = -(1.0 / counts_sum**2) * dcounts_sum_inv
cmp("counts_sum", dcounts_sum, counts_sum)

dcounts = (counts_sum_inv * dprobs) + torch.ones_like(counts) * dcounts_sum
cmp("counts", dcounts, counts)

dnorm_logits = dcounts * counts
cmp("norm_logits", dnorm_logits, norm_logits)
dlogit_maxes = -dnorm_logits.sum(1, keepdim=True)
cmp("logit_maxes", dlogit_maxes, logit_maxes)

dlogits = (
    dnorm_logits.clone()
    + F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
)
cmp("logits", dlogits, logits)
dh = dlogits @ W2.T
cmp("h", dh, h)
dW2 = h.T @ dlogits
cmp("W2", dW2, W2)
db2 = dlogits.sum(0)
cmp("b2", db2, b2)
dhpreact = (1.0 - h**2) * dh
cmp("hpreact", dhpreact, hpreact)
# Note: always be careful about broadcasting

dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
cmp("bngain", dbngain, bngain)
dbnraw = bngain * dhpreact
cmp("bnraw", dbnraw, bnraw)
dbnbias = dhpreact.sum(0, keepdim=True)
cmp("bnbias", dbnbias, bnbias)
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
cmp("bnvar_inv", dbnvar_inv, bnvar_inv)
dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
cmp("bnvar", dbnvar, bnvar)
dbndiff2 = torch.ones_like(bndiff2) * (1.0 / (n - 1)) * dbnvar
cmp("bndiff2", dbndiff2, bndiff2)
dbndiff = bnvar_inv * dbnraw + 2 * bndiff * dbndiff2
cmp("bndiff", dbndiff, bndiff)
dbnmeani = -dbndiff.sum(0, keepdim=True)
cmp("bnmeani", dbnmeani, bnmeani)
dhprebn = (1.0 / n) * torch.ones_like(hprebn) * dbnmeani + dbndiff.clone()
cmp("hprebn", dhprebn, hprebn)
dembcat = dhprebn @ W1.T
cmp("embcat", dembcat, embcat)
dW1 = embcat.T @ dhprebn
cmp("W1", dW1, W1)
db1 = dhprebn.sum(0)
cmp("b1", db1, b1)
demb = dembcat.view(emb.shape)
cmp("emb", demb, emb)
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]
cmp("C", dC, C)

# %% Cell 9

# Exercise 2: backprop through cross entropy but all in one go

loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), "diff: ", (loss_fast - loss).item())
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
cmp("logits", dlogits, logits)

# %% Cell 10

# Exercise 3: backprop through batchnorm in one go

hpreact_fast = (
    bngain
    * (hprebn - hprebn.mean(0, keepdim=True))
    / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5)
    + bnbias
)
print("max diff: ", (hpreact_fast - hpreact).abs().max().item())
dhprebn = (
    bngain
    * bnvar_inv
    / n
    * (n * dhpreact - dhpreact.sum(0) - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0))
)
cmp("hprebn", dhprebn, hprebn)

# %% Cell 11

# init
n_embd = 10  # the dimensionality of the character embedding vectors
n_hidden = 200  # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)  # for reproducibility
C = torch.randn((vocab_size, n_embd), generator=g)
# Layer 1
W1 = (
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / ((n_embd * block_size) ** 0.5)
)
b1 = torch.randn(n_hidden, generator=g) * 0.1
# Layer 2
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters))  # number of parameters in total
for p in parameters:
    p.requires_grad = True

# same optimization as last time
max_steps = 200000
batch_size = 32
n = batch_size  # convenience
lossi = []

with torch.no_grad():
    # kick off optimization
    for i in range(max_steps):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]  # batch X,Y

        # forward pass
        emb = C[Xb]  # embed the characters into vectors
        embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors
        # Linear layer
        hprebn = embcat @ W1 + b1  # hidden layer pre-activation
        # BatchNorm layer
        # -------------------------------------------------------------
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5) ** -0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        # -------------------------------------------------------------
        # Non-linearity
        h = torch.tanh(hpreact)  # hidden layer
        logits = h @ W2 + b2  # output layer
        loss = F.cross_entropy(logits, Yb)  # loss function

        # backward pass
        for p in parameters:
            p.grad = None
        # loss.backward() # use this for correctness comparisons, delete it later!

        # manual backprop! #swole_doge_meme
        # -----------------
        # YOUR CODE HERE :)
        dlogits = F.softmax(logits, 1)
        dlogits[range(Xb.shape[0]), Yb] -= 1
        dlogits /= n

        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)

        dhpreact = (1.0 - h**2) * dh

        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dhprebn = (
            bngain
            * bnvar_inv
            / n
            * (
                n * dhpreact
                - dhpreact.sum(0)
                - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
            )
        )

        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)

        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for k in range(Xb.shape[0]):
            for j in range(Xb.shape[1]):
                ix = Xb[k, j]
                dC[ix] += demb[k, j]
        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
        # -----------------

        # update
        lr = 0.1 if i < 100000 else 0.01  # step learning rate decay
        for p, grad in zip(parameters, grads):
            # p.data += (
            #    -lr * p.grad
            # )  # old way of cheems doge (using PyTorch grad from .backward())
            p.data += -lr * grad  # new way of swole doge TODO: enable

        # track stats
        if i % 10000 == 0:  # print every once in a while
            print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
        lossi.append(loss.log10().item())


# %% Cell 12


with torch.no_grad():
    # pass the training set through
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    # measure the mean/std over the entire training set
    bnmean = hpreact.mean(0, keepdim=True)
    bnvar = hpreact.var(0, keepdim=True, unbiased=True)

# %% Cell 13


@torch.no_grad()  # this decorator disables gradient tracking
def split_loss(split):
    x, y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte),
    }[split]
    emb = C[x]  # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1)  # concat into (N, block_size * n_embd)
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5) ** -0.5 + bnbias
    h = torch.tanh(hpreact)  # (N, n_hidden)
    logits = h @ W2 + b2  # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


split_loss("train")
split_loss("val")
