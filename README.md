# pokepy :)

![Alt text](pokemon.jpg) </br>

a character-level language model built completely from scratch using only numpy that generates pokémon sounding names. no PyTorch, no autograd, no deep learning frameworks — every forward pass, backward pass, and gradient update is manually implemented.

this project started as a way to understand how neural networks actually work under the hood. I first built a simple MLP, then expanded it into a WaveNet-style architecture to explore how increasing context length changes what a model can learn.

## demo

try it live (local desktop):

for mac users, double click PokepyLauncher.command to launch the application.
for windows users, double click PokepyLauncher.bat to launch the application.

once running, the terminal will spin up a local matrix-inference engine. open your browser and navigate to: http://localhost:10000

---

# what is this

I wanted to understand the foundations behind language models, so I built a mini character-level text generator completely from scratch.

instead of relying on existing machine learning libraries, I manually implemented:

* embeddings
* linear layers
* batch normalization
* tanh activations
* softmax
* cross entropy loss
* backpropagation
* gradient descent

everything runs only with numpy.

the model learns character patterns from pokémon names and generates new names based on the relationships it discovers.

---

# model 1 — MLP

## architecture

the first version was a simple multilayer perceptron:

* character embedding layer (10-dimensional vectors)
* linear layer
* batch normalization
* tanh activation
* output linear layer
* softmax + cross entropy loss

hidden size:

```
200 neurons
```

context length:

```
3 characters
```

this means the model only looks at the previous 3 characters to predict the next one.

example:

```
pik → a
ika → next character
```

the context continuously shifts as the model generates.

---

## MLP results

training:

```
train loss: 1.294
validation loss: 3.504
```

generated names:

```
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
```

the MLP learned basic character relationships, but the limited context window made it difficult to understand longer patterns inside names.

---

# model 2 — WaveNet

## why I built this

the biggest limitation of the MLP was context length.

with only 3 characters of context, the model could only see a small part of each name.

for example:

```
charizard

cha
har
ari
riz
```

the model does not understand the larger structure of the word.

WaveNet improves this by gradually combining groups of characters, allowing the model to build larger representations without massively increasing the number of parameters.

---

## architecture

WaveNet-style architecture:

* character embedding layer (10 dimensions)
* FlattenConsecutive layers
* multiple linear layers
* batch normalization
* tanh activations
* final output layer
* softmax + cross entropy loss

context length:

```
8 characters
```

the model builds information hierarchically:

```
characters

↓

combined character groups

↓

higher level features

↓

next character prediction
```

---

## WaveNet results

training:

```
train loss: 1.949
validation loss: 2.748
```

generated names:

```
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
```

WaveNet produced longer and more structured generations because it had access to a larger context window.

---

# MLP vs WaveNet

|                  | MLP                 | WaveNet             |
| ---------------- | ------------------- | ------------------- |
| Context size     | 3 characters        | 8 characters        |
| Hidden size      | 200                 | 32                  |
| Training steps   | 300,000             | 10,000              |
| Architecture     | Single hidden layer | Hierarchical layers |
| Feature learning | Direct              | Progressive         |
| Main advantage   | Simple baseline     | Larger context      |

---

# challenges

## context length

one of the biggest lessons from this project was understanding why context matters.

a model with a smaller context window can only learn local patterns, while larger context allows it to understand longer relationships.

the MLP used:

```
3 character context
```

while WaveNet increased this to:

```
8 character context
```

which allowed it to capture more structure from names.

---

## batch normalization

implementing batch normalization manually was one of the hardest parts.

I had to handle:

* batch mean
* batch variance
* running mean
* running variance

training and inference use different statistics, so saving the running values was required for the deployed model to generate correctly.

---

## backpropagation

instead of using:

```python
loss.backward()
```

I manually calculated gradients for:

* embeddings
* linear layers
* batch normalization
* tanh activations
* softmax cross entropy

this helped me understand how neural networks actually learn instead of treating them as black boxes.

---

## random generation

generation is probabilistic.

even with the same trained model, outputs change because the next character is sampled from the model's probability distribution.

---

# training details

## MLP

dataset:

```
pokemon names
```

training:

* optimizer: SGD
* batch size: 32
* steps: 300,000
* learning rate: `0.1 → 0.01 after 100k steps`

parameters:

```
C
W1
W2
b2
bngain
bnbias
```

---

## WaveNet

dataset:

```
pokemon names
```

training:

* optimizer: SGD
* batch size: 32
* steps: 10,000
* learning rate: `0.1 → 0.01 after 8000 steps`

parameters:

```
embeddings
linear layers
batch normalization parameters
```

---

# deployment

the model is deployed using Hugging Face Spaces with Gradio.

the demo loads the trained numpy weights and runs inference without PyTorch or external ML frameworks.

the deployed model uses:

* trained embeddings
* linear layer weights
* batch normalization parameters
* vocabulary mappings

the entire inference pipeline runs using manually implemented numpy layers.

---

# usage

install dependencies:

```bash
pip install numpy
```

run:

```bash
python mlp.py
python wavenet.py
```

you will need:

```
data/

└── pokemon.txt
```

with one pokémon name per line.

---

# what I learned

this project taught me how language models are built from the ground up.

the biggest takeaway was that improving a model is not always about making it bigger. changing the architecture and giving the model better ways to understand context can have a larger impact than simply adding more parameters.

---

# license

MIT

heavily inspired by Andrej Karpathy's makemore series — highly recommend if you want to understand neural networks from the inside out :)
