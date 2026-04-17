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


"""
Notes:

1. Fixing softmax
We see that the starting loss is really high, even though we can estimate that it should be around 3.3
because that would be equivalent to assigning the same confidence to each value, so basically random chance
This happens because we just let our W2 and b2 be initialised at fully random, generating false confidence and wasting training
Instead, we should normalize them
Also, we do not want to start at 0, because then we lack entropy and the network might struggle to train

2. Fixing the tanh saturation
Same thing but happening with pre-activation weights init at random
They are too big and they saturate the tanh to -1 or 1. But that means
that in backprop the gradient gets 0, so there is no gradient flowing there
We do not want that to be the case at init time, because in the worst case we could have a neuron that is fully saturated and therefore just dead
So we also need to normalize the weights of w1, b1. We want things to be roughly unit gaussian.

3. Usual init solution -> Kaiming normal, dividing by sqrt(fan_in) -> proposed by kaimin he\
This is no longer that much of a fundamental thing because nn are more stable because of things like:
    normalization layers, better optimizers, ...

4. batch normalization
We can normalize the pre-activations in order to make them unit gaussian
Doing this alone is not great for training, because we want to give the nn freedom to be more spiky or more spread
in order for it to be more expressive. To achieve that we use a sclae and shift parameters.
We can init them as torch ones and zeros of size (1, n) where n is the number of neurons in the layer (hidden_dim)
and hpreact becomes gain((hpreact - mean) / std) + bias
gain and bias get backpropagated, that way training can scaled the distribution
-> An interesting side effect of batch norm is that we now introduce a link between elements in a batch
indeed they are now normalised as a function of teh batch and this ends up being a form of regularization.
But this has also made it so that people explore other forms of normalization that dont have this side effect.\
 -> we also remove the bias from the layer when we do batch norm, it is useless because it is removed by the mean
 and we introduce the batch norm bias
 Momentum is used to estimate the final batch nmorm values during training, it is important that if batches are too small it is config properly

5. training observability tricks
Look at the distribution of you activations and weight gradients
Look at the ratio between your weights and the gradient, it should be around 1e-3, this ensure it trains butr not too abruptly


"""
