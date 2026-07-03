# pokepy :)

![Alt text](pokemon.jpg)

A character-level language model built completely from scratch using only NumPy that generates Pokémon-sounding names. No PyTorch, no autograd, no deep learning frameworks — every forward pass, backward pass, and gradient update is manually implemented.

This project started as a way to understand how neural networks actually work under the hood. I first built a simple MLP, then expanded it into a WaveNet-style architecture to explore how increasing context length changes what a model can learn.

## Demo

This project is fully distributed on PyPI. Follow these steps to set up an isolated sandbox environment and run the demo across macOS, Linux, or Windows.

### Initialize a venv

To bypass native system file locks on macOS and ensure dependencies install safely, create and activate a localized virtual environment:

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```dos
python -m venv .venv
.venv\Scripts\activate
```

### Install the package

```bash
pip install pokepy-generator
```

### Launch the app

Run the terminal execution shortcut from anywhere in the activated environment:

```bash
pokepy
```

Then open your browser and head to the local link:

```
http://127.0.0.1:7860
```

## What is this?

I wanted to understand the foundations behind language models, so I built a mini character-level text generator completely from scratch.

Instead of relying on existing machine learning libraries, I manually implemented:

- embeddings
- linear layers
- batch normalization
- tanh activations
- softmax
- cross entropy loss
- backpropagation
- gradient descent

Everything runs only with NumPy. The model learns character patterns from Pokémon names and generates new names based on the relationships it discovers.

## Model 1 — MLP

### Architecture

The first version was a simple multilayer perceptron:

- character embedding layer (10-dimensional vectors)
- linear layer
- batch normalization
- tanh activation
- output linear layer
- softmax + cross entropy loss

**Hidden size:** 200 neurons
**Context length:** 3 characters

This means the model only looks at the previous 3 characters to predict the next one.

Example:
```
pik → a
ika → next character
```

The context continuously shifts as the model generates.

### MLP results

**Training:**
- train loss: 1.294
- validation loss: 3.504

**Generated names:**
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

The MLP learned basic character relationships, but the limited context window made it difficult to understand longer patterns inside names.

## Model 2 — WaveNet

### Why I built this

The biggest limitation of the MLP was context length. With only 3 characters of context, the model could only see a small part of each name.

For example, `charizard`:
```
cha
har
ari
riz
```

The model does not understand the larger structure of the word.

WaveNet improves this by gradually combining groups of characters, allowing the model to build larger representations without massively increasing the number of parameters.

### Architecture

WaveNet-style architecture:

- character embedding layer (10 dimensions)
- FlattenConsecutive layers
- multiple linear layers
- batch normalization
- tanh activations
- final output layer
- softmax + cross entropy loss

**Context length:** 8 characters

The model builds information hierarchically:

```
characters
    ↓
combined character groups
    ↓
higher level features
    ↓
next character prediction
```

### WaveNet results

**Training:**
- train loss: 1.949
- validation loss: 2.748

**Generated names:**
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

## MLP vs WaveNet

| | MLP | WaveNet |
|---|---|---|
| Context size | 3 characters | 8 characters |
| Hidden size | 200 | 32 |
| Training steps | 300,000 | 10,000 |
| Architecture | Single hidden layer | Hierarchical layers |
| Feature learning | Direct | Progressive |
| Main advantage | Simple baseline | Larger context |

## Challenges

### Context length

One of the biggest lessons from this project was understanding why context matters. A model with a smaller context window can only learn local patterns, while a larger context allows it to understand longer relationships.

The MLP used a 3-character context, while WaveNet increased this to an 8-character context, which allowed it to capture more structure from names.

### Batch normalization

Implementing batch normalization manually was one of the hardest parts. I had to handle:

- batch mean
- batch variance
- running mean
- running variance

Training and inference use different statistics, so saving the running values was required for the deployed model to generate correctly.

### Backpropagation

Instead of using:
```python
loss.backward()
```

I manually calculated gradients for:

- embeddings
- linear layers
- batch normalization
- tanh activations
- softmax cross entropy

This helped me understand how neural networks actually learn instead of treating them as black boxes.

### Random generation

Generation is probabilistic. Even with the same trained model, outputs change because the next character is sampled from the model's probability distribution.

## Training details

### MLP

**Dataset:** Pokémon names

**Training:**
- optimizer: SGD
- batch size: 32
- steps: 300,000
- learning rate: 0.1 → 0.01 after 100k steps

**Parameters:**
- `C`
- `W1`
- `W2`
- `b2`
- `bngain`
- `bnbias`

### WaveNet

**Dataset:** Pokémon names

**Training:**
- optimizer: SGD
- batch size: 32
- steps: 10,000
- learning rate: 0.1 → 0.01 after 8000 steps

**Parameters:**
- embeddings
- linear layers
- batch normalization parameters

## Deployment

The package mounts a local Gradio interface using explicit block structural routing to circumvent downstream context caching errors.

The demo loads the trained NumPy weights locally and runs inference instantly without requiring external ML frameworks.

The architecture bundles:

- trained embeddings
- linear layer weights
- batch normalization parameters
- vocabulary mappings

The entire inference pipeline runs using manually implemented NumPy layers.

## License

MIT

Heavily inspired by Andrej Karpathy's makemore series — highly recommend it if you want to understand neural networks from the inside out :)