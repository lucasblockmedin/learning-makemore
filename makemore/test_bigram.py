# %% Cell 1
words = open("names.txt", "r").read().split()

# %% Cell 2
b = {}
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1


# %% Cell 3
sorted(b.items(), key=lambda kv: -kv[1])


# %% Cell 4
import torch

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

print(N)

# %% Cell 5
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="center", color="grey")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="grey")
plt.axis("off")
plt.show()

# %% Cell 6

P = (N + 1).float()  # Model smoothing, there is no impossible prob
P /= P.sum(1, keepdim=True)  # NOTE: be careful about broadcasting!

g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))


# %% Cell 7

log_likelihood = 0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{ch1}{ch2}: prob={prob:.4f}, logprob={logprob:.4f}")
print(log_likelihood)
nll = -log_likelihood
print(nll / n)


# %% Cell 8
# Create the training set for our nn

xs, ys = [], []

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(xs.nelement())

# %% Cell 9
W = torch.randn((27, 27), generator=g, requires_grad=True)

# %% Cell 10
import torch.nn.functional as F

for k in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    # forward
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(
        1, keepdim=True
    )  # equivalent to F.softmax(logits, dim=-1)
    loss = (
        -probs[torch.arange(len(xs)), ys].log().mean() + 0.01 * (W**2).mean()
    )  # regularization
    print(loss.item())
    # backward
    W.grad = None  # set grad to zero
    loss.backward()
    # update the weights
    W.data += -10 * W.grad


"""
SOME NOTES:
The Representation: Logits (\(z\)) are the raw output scores of a model; they represent the log-counts of each class in the dataset.
The Transformation: Softmax converts these logits into probabilities (\(P\)) by exponentiating them (\(e^z\)) and normalizing the total to 1.
The Justification: Softmax is the unique "Maximum Entropy" function, meaning it turns scores into probabilities without making up extra information.
The Physics Link: This follows the Boltzmann Distribution, which is the most "natural" way to distribute energy (or probability) across different states.
The Training Goal: Using NLL Loss forces the model to adjust its weights until the exponentiated logits match the actual frequencies found in the data.

Parameter Equivalence: A single-layer neural network with a 27x27 weight matrix has exactly the same number of parameters as a 27x27 counting table.
Linear Mapping: Because the model only looks at the current character (one-hot encoded) to predict the next, it can only learn pairwise relationships—it has no "memory" of earlier characters.
NLL Floor: The counting table represents the "perfect" empirical distribution of the data; a single layer can only aim to replicate those exact probabilities.
Memory Bottleneck: To achieve a lower NLL than a bigram, a model must increase its context (e.g., using a hidden layer or multiple previous characters) to reduce the data's entropy.

Regularization pulls the weights towards smoothing, if the probs get too big the regularization term dominates
"""
