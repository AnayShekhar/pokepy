import re
import numpy as np # using only numpy :)
import os

# read .txt and clean 
path = os.path.join(os.path.dirname(__file__), '..', 'data', 'pokemon.txt')
words = open(path).read().splitlines()
words = [re.sub(r'[^a-z]', '', w.lower()) for w in words]
words = [w for w in words if len(w) > 0]

# shuffle 
rng = np.random.RandomState(67)
rng.shuffle(words)

# build vocab
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)

# build the dataset
block_size = 8
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

    return np.array(X), np.array(Y)
    
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

# linear layer
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = np.random.randn(fan_in, fan_out) / (fan_in**0.5)
        self.bias = np.zeros((1, fan_out)) if bias else None

    def __call__(self, x):
        self.x = x
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def backward(self, dout):
        self.dx = dout @ self.weight.T
       
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        dout_flat = dout.reshape(-1, dout.shape[-1])
        self.dweight = x_flat.T @ dout_flat

        if self.bias is not None:
            self.dbias = dout_flat.sum(axis=0, keepdims=True)
        return self.dx 

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

# batchnorm layer
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))
        
    def __call__(self, x):
        self.x_shape = x.shape
        if len(x.shape) == 3:
            x = x.reshape(-1, x.shape[-1])
        
        if self.training:
            self.xmean = x.mean(axis=0, keepdims=True)
            self.xvar = x.var(axis=0, keepdims=True)
            self.xhat = (x - self.xmean) / np.sqrt(self.xvar + self.eps)
            self.out = self.gamma * self.xhat + self.beta
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.xvar
        else:
            xhat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.out = self.gamma * xhat + self.beta
            
        if len(self.x_shape) == 3:
            self.out = self.out.reshape(self.x_shape)
        return self.out

    def backward(self, dout):
        dout_orig_shape = dout.shape
        if len(dout.shape) == 3:
            dout = dout.reshape(-1, dout.shape[-1])
            
        N = dout.shape[0]
        
        self.dgamma = np.sum(dout * self.xhat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        
        dxhat = dout * self.gamma
        self.dx = (1. / N) / np.sqrt(self.xvar + self.eps) * (
            N * dxhat - np.sum(dxhat, axis=0, keepdims=True) - 
            self.xhat * np.sum(dxhat * self.xhat, axis=0, keepdims=True)
        )
        return self.dx.reshape(dout_orig_shape)

    def parameters(self):
        return [self.gamma, self.beta]

# tanh activation function
class Tanh:
    def __call__(self, x):
        self.out = np.tanh(x)
        return self.out
        
    def backward(self, dout):
        return dout * (1 - self.out**2)

    def parameters(self):
        return []

# embedding layer
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = np.random.randn(num_embeddings, embedding_dim)

    def __call__(self, IX):
        self.IX = IX
        self.out = self.weight[IX]
        return self.out

    def backward(self, dout):
        self.dweight = np.zeros_like(self.weight)
        np.add.at(self.dweight, self.IX, dout)
        return None

    def parameters(self):
        return [self.weight]

# combines consecutive layers (wavenet)
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
        
    def __call__(self, x):
        self.input_shape = x.shape
        B, T, C = x.shape
        x = x.reshape(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.reshape(B, -1)
        self.out = x
        return self.out

    def backward(self, dout):
        if len(dout.shape) == 2:
            dout = dout[:, np.newaxis, :]
        return dout.reshape(self.input_shape)

    def parameters(self):
        return []

# runs layers in order
class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers: 
            x = layer(x)
        self.out = x
        return self.out
        
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
        
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
# parameters
n_embd = 10
n_hidden = 32

np.random.seed(67)

emb_layer = Embedding(vocab_size, n_embd)

model = Sequential([
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False),  BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

for layer in model.layers[:-1]:
    if isinstance(layer, Linear):
        layer.weight *= 5/3

# make final layer less confident
model.layers[-1].weight *= 0.1
parameters = [emb_layer] + model.layers

# training loop
max_steps = 10000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch construct
    ix = np.random.randint(0, Xtr.shape[0], batch_size) 
    Xb, Yb = Xtr[ix], Ytr[ix]
    # forward pass
    emb = emb_layer(Xb)
    logits = model(emb)
    # cross entropy loss
    logits = logits - logits.max(axis=1, keepdims=True)
    counts = np.exp(logits)
    probs = counts / counts.sum(axis=1, keepdims=True)
    loss = -np.log(probs[range(batch_size), Yb]).mean()
    # backward pass
    dlogits = probs.copy()
    dlogits[range(batch_size), Yb] -= 1
    dlogits /= batch_size
    demb = model.backward(dlogits)
    emb_layer.backward(demb)
    # update
    lr = 0.1 if i < 8000 else 0.01 # learning rate decay
    for p in parameters:
        if isinstance(p, Linear):
            p.weight[...] -= lr * p.dweight
            if p.bias is not None:
                p.bias[...] -= lr * p.dbias
        elif isinstance(p, BatchNorm1d):
            p.gamma[...] -= lr * p.dgamma
            p.beta[...] -= lr * p.dbeta
        elif isinstance(p, Embedding):
            p.weight[...] -= lr * p.dweight
    # track stats
    if i % 1000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss:.4f}')

    lossi.append(np.log10(loss))

# evaluate the loss
for layer in model.layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val':   (Xdev, Ydev),
        'test':  (Xte, Yte),
    }[split]

    emb = emb_layer(x)
    logits = model(emb)
    logits -= logits.max(axis=1, keepdims=True)
    counts = np.exp(logits)
    probs = counts / counts.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(y)), y]).mean()
    return loss
    
print('train', split_loss('train'))
print('val', split_loss('val'))

# sample from model

for layer in model.layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False
        
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = emb_layer(np.array([context]))
        logits = model(emb)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        ix = np.random.choice(vocab_size, p=probs[0])
        context = context[1:] + [ix]
        out.append(ix)

        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

