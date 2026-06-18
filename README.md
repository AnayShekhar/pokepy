# pokepy :)
![Alt text](pokemon.jpg) </br>

a character-level language model built completely from scratch using only numpy that generates pokémon sounding names. no PyTorch, no autograd, no frameworks — every forward pass, backward pass, and gradient update is manually implemented.

this project started as a way to understand how neural networks actually work under the hood. I first built a simple MLP, then improved it into a WaveNet-style architecture to understand how increasing context length changes what the model can learn.

---

## what is this

I wanted to understand the foundations behind language models, so I built a mini character-level text generator from scratch.

instead of using existing deep learning libraries, I implemented:

- embeddings
- linear layers
- batch normalization
- tanh activations
- softmax
- cross entropy loss
- backpropagation
- gradient descent updates

everything runs only with numpy.

the model learns patterns from pokémon names and generates completely new names based on the character relationships it learned.

---

# model 1 — MLP

## architecture

the first version was a simple multilayer perceptron:

- character embedding layer (10-dimensional vectors)
- linear layer
- batch normalization
- tanh activation
- output linear layer
- softmax + cross entropy loss

hidden size: `200 neurons`

context length: `3 characters`

meaning the model only looks at the previous 3 characters to predict the next character.

example: `pik` predicts `a`, then the context shifts: `ika` and predicts the next character.

---

## MLP results

training:

train loss: 1.294
validation loss: 3.504

generated names:

blipedeedo
rosalini
lect
dartic
star
vigus
swannon
hippowdon
the
larvinerao

the MLP was able to learn basic character patterns, but because the context window was only 3 characters, it struggled with longer dependencies.

---

# model 2 — WaveNet

## why I built this

the biggest limitation of the MLP was context length.
with only 3 characters, the model could only see a tiny part of each name.

for example, `charizard` — the MLP only sees:

cha

har

ari

riz

instead of understanding the entire structure.
WaveNet fixes this by gradually combining character groups, allowing the model to see a larger context without making the network extremely large.

---

## architecture

WaveNet style model:

- character embedding layer (10 dimensions)
- FlattenConsecutive layers
- multiple linear layers
- batch normalization
- tanh activations
- final output layer
- softmax + cross entropy

context length: `8 characters`

the model slowly compresses information:
8 characters

↓

combine pairs

↓

larger features

↓

predict next character

---

## WaveNet results

training:
train loss: 1.949

validation loss: 2.747

generated names:
gropinig

pyghislacat

poloun

hoongel

spuspiniyan

ongtover

kasato

xel

felspipon

linmatie

asherron

beatdiqdule

madstutf

drudona

rouzslra

liwsywunk

galeon

magnoslaws

araidono

lickopt

the WaveNet generated longer and more structured names because it had a larger context window.

---

# MLP vs WaveNet

| | MLP | WaveNet |
|---|---|---|
| Context size | 3 characters | 8 characters |
| Hidden size | 200 | 32 |
| Training steps | 300,000 | 10,000 |
| Architecture | Single hidden layer | Hierarchical layers |
| Parameters | Larger | More efficient |
| Generation | Shorter patterns | Longer structures |

---

# things I struggled with

## context length

one of the biggest lessons from this project was understanding why context matters.

a model with a small context window can only learn local patterns, but struggles with longer relationships.

in the MLP: `context = 3 characters`

the model mostly learned small patterns between characters. WaveNet improved this by giving the model access to more previous characters.

---

## batch normalization

implementing batch normalization manually was one of the hardest parts.

I had to track:

- batch mean
- batch variance
- running mean
- running variance

because training and inference behave differently.

---

## backpropagation

instead of using:

```python
loss.backward()
```

I manually calculated gradients for every layer.

this helped me understand how neural networks actually learn instead of treating them like a black box.

---

## random generation

generation is probabilistic. even with the same trained model, every run can create different names because the next character is sampled from the probability distribution.

---

# training details

## MLP

- dataset: pokémon names
- optimizer: SGD
- batch size: 32
- steps: 300,000
- learning rate: `0.1 → 0.01 after 100k steps`

parameters: `C`, `W1`, `W2`, `b2`, `bngain`, `bnbias`

## WaveNet

- dataset: pokémon names
- optimizer: SGD
- batch size: 32
- steps: 10,000
- learning rate: `0.1 → 0.01 after 8000 steps`

parameters: `embeddings`, `linear layers`, `batchnorm parameters`

---

# usage

```bash
pip install numpy
python mlp.py
python wavenet.py
```

you'll need:
data/

└── pokemon.txt

with one pokémon name per line.

---

# what I learned

this project taught me how language models are built from the ground up.

the biggest takeaway was that improving a model is not always about making it bigger — changing the architecture and giving it better ways to understand context can make a huge difference.

---

# license

MIT

heavily inspired by Andrej Karpathy's makemore series — highly recommend if you want to understand neural networks from the inside out :)