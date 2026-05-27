import re
import numpy as np # using only numpy :)

# read .txt and clean 
words = open('pokemon.txt').read().splitlines()
words = [re.sub(r'[^a-z]', '', w.lower()) for w in words]
words = [w for w in words if len(w) > 0]

# build vocab
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)

# build the dataset
block_size = 3 
X, Y = [], []
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] 
            
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    return X, Y

X, Y = build_dataset(words)

# parameters
n_embd = 10
n_hidden = 200

np.random.seed(67)
C  = np.random.randn(vocab_size, n_embd)
W1 = np.random.randn(n_embd * block_size, n_hidden) * (5/3)/((n_embd * block_size)**0.5)
W2 = np.random.randn(n_hidden, vocab_size) * 0.1
b2 = np.random.randn(vocab_size) * 0.1
bngain = np.random.randn(1, n_hidden) * 0.1 + 1.0
bnbias = np.random.randn(1, n_hidden) * 0.1
bn_mean_running = np.zeros((1, n_hidden))
bn_var_running = np.ones((1, n_hidden))
parameters = [C, W1, W2, b2, bngain, bnbias]

# training loop
max_steps = 300000
batch_size = 32
n = batch_size
lossi = []
for i in range(max_steps):
    # minibatch constuction
    ix = np.random.randint(0, X.shape[0], batch_size)
    Xb, Yb = X[ix], Y[ix]

    # forward pass
    # linear layer 1
    emb = C[Xb]
    embcat = emb.reshape(emb.shape[0], -1)
    hprebn = embcat @ W1 
    bnmeani = hprebn.mean(0, keepdims=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    bnvar = 1/(n-1)*(bndiff2).sum(0, keepdims=True) 
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bn_mean_running = 0.999 * bn_mean_running + 0.001 * bnmeani
    bn_var_running = 0.999 * bn_var_running + 0.001 * bnvar
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # Non-linearity
    h = np.tanh(hpreact) # hidden layer
    # Linear layer 2
    logits = h @ W2 + b2 # output layer
    # cross entropy loss (same as F.cross_entropy(logits, Yb))
    logit_maxes = logits.max(1, keepdims=True)
    norm_logits = logits - logit_maxes # subtract max for numerical stability
    counts = np.exp(norm_logits)
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum**-1
    probs = counts * counts_sum_inv
    logprobs = np.log(probs)
    loss = -logprobs[range(n), Yb].mean()

    # backward pass
    dlogprobs = np.zeros_like(logprobs)
    dlogprobs[range(n), Yb] = -1.0/n
    dprobs = dlogprobs / probs
    dcounts_sum_inv = (counts * dprobs).sum(1, keepdims=True)
    dcounts = counts_sum_inv * dprobs + np.ones_like(counts) * (-counts_sum**-2) * dcounts_sum_inv
    dlogits = counts * dcounts
    dlogits += np.eye(logits.shape[1])[logits.argmax(1)] * (-dlogits.sum(1, keepdims=True))
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    dhpreact = (1.0 - h**2) * dh
    dbngain = (bnraw * dhpreact).sum(0, keepdims=True)
    dbnbias = dhpreact.sum(0, keepdims=True)
    dbnraw = bngain * dhpreact
    dbndiff = bnvar_inv * dbnraw + (2*bndiff) * ((1.0/(n-1)) * np.ones_like(bndiff2) * ((-0.5*(bnvar + 1e-5)**-1.5) * (bndiff * dbnraw).sum(0, keepdims=True)))
    dhprebn = dbndiff - dbndiff.sum(0, keepdims=True)/n
    dW1 = embcat.T @ dhprebn
    db2_unused = dhprebn.sum(0)
    dembcat = dhprebn @ W1.T
    demb = dembcat.reshape(emb.shape)
    dC = np.zeros_like(C)
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            dC[Xb[k,j]] += demb[k,j]

    grads = [dC, dW1, dW2, db2, dbngain, dbnbias]

    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p, grad in zip(parameters, grads):
        p[...] = p - lr * grad
    # track 
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {float(loss):.4f}')
        lossi.append(float(np.log10(loss)))


# sample from the model
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        emb = C[np.array([context])]
        embcat = emb.reshape(1, -1)
        hprebn = embcat @ W1
        bnraw = (hprebn - bn_mean_running) / np.sqrt(bn_var_running + 1e-5)
        hpreact = bngain * bnraw + bnbias
        h = np.tanh(hpreact)
        logits = h @ W2 + b2
        logits -= logits.max(1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(1, keepdims=True)
        ix = np.random.choice(vocab_size, p=probs[0])
        context = context[1:] + [ix]
        if ix == 0 or len(out) >= 10:  # stop at 10 characters
            break
        out.append(itos[ix])
    print(''.join(out))