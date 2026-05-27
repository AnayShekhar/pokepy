# pokepy :)

a 2 layer MLP built from scratch using only numpy that generates pokémon-sounding names. no PyTorch, no autograd, every forward pass, backward pass, and gradient update is written by hand.

## what is this

I wanted to really understand how neural networks work under the hood, so I built one from scratch using only numpy. no autograd, no `.backward()`, every single gradient is derived and computed manually, including backprop through batchnorm.
trained on a dataset of pokémon names, the model learns character patterns and generates new names that sound like they could actually be pokémon.

## architecture

- character embedding layer (10-dimensional vectors)
- linear layer → batch normalization → tanh (200 hidden neurons)
- linear layer → softmax
- cross entropy loss

block size of 3 — meaning it looks at the last 3 characters to predict the next one. running mean and variance are tracked during training so batchnorm works correctly at inference time.

## some generated names
marina </br>
panny </br>
maa </br>
cetill </br>
magmoon </br>
artroseuli </br>
nown </br>
shedicotto </br>
pie </br>
azentas </br>

## training details

- 300,000 training steps with minibatches of 32
- learning rate starts at 0.1, decays to 0.01 at step 100,000
- final loss around 1.4–1.7
- parameters: C (embeddings), W1, W2, b2, bngain, bnbias

## usage

```bash
pip install numpy
python pokepy.py
```

you'll need a `pokemon.txt` with one pokémon name per line.

## inpiration
heavily inspired by andrej karpathy's makemore series, highly recommend if you want to actually understand how this stuff works :)
