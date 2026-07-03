import gradio as gr
import numpy as np
import os
import sys
import re
import importlib.resources  # Native, standard library wrapper

# Set seed for reproducibility
np.random.seed(67)

# ------------------------------------------------------------------------------
# Pure NumPy Architectural Primitives
# ------------------------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = np.zeros((fan_in, fan_out))
        self.bias = np.zeros((1, fan_out)) if bias else None

    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out


class BatchNorm1d:
    def __init__(self, dim):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))

    def __call__(self, x):
        xhat = (x - self.running_mean) / np.sqrt(self.running_var + 1e-5)
        return self.gamma * xhat + self.beta


class Tanh:
    def __call__(self, x):
        return np.tanh(x)


class Embedding:
    def __init__(self, vocab_size, emb_dim):
        self.weight = np.zeros((vocab_size, emb_dim))

    def __call__(self, x):
        return self.weight[x]


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.reshape(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.reshape(B, -1)
        return x


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ------------------------------------------------------------------------------
# Robust Asset Path Resolution (Supports PyInstaller, PyPI Modern Installs)
# ------------------------------------------------------------------------------
def get_asset_path(filename):
    """ Resolves absolute paths safely across dev environments, PyInstaller, and pip distribution """
    # 1. Check for PyInstaller runtime unpack folder
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    
    # 2. Direct fallback to checking adjacent to the file location itself
    # This works flawlessly for flat modules installed via setuptools
    base_path = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base_path, filename)
    if os.path.exists(local_path):
        return local_path
        
    # 3. Structural fallback via importlib
    try:
        ref = importlib.resources.files("pokepy").joinpath(filename)
        if ref.exists():
            return str(ref)
    except Exception:
        pass
        
    return local_path


# ------------------------------------------------------------------------------
# Vocabulary & Dataset Extraction Setup
# ------------------------------------------------------------------------------
words = open(get_asset_path("pokemon.txt")).read().splitlines()
words = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
words = [w for w in words if len(w) > 0]

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
block_size = 8


# ------------------------------------------------------------------------------
# Model Pipeline Instantiation & Parameter Injection
# ------------------------------------------------------------------------------
model = Sequential([
    FlattenConsecutive(2),
    Linear(20, 32, False),
    BatchNorm1d(32),
    Tanh(),

    FlattenConsecutive(2),
    Linear(64, 32, False),
    BatchNorm1d(32),
    Tanh(),

    FlattenConsecutive(2),
    Linear(64, 32, False),
    BatchNorm1d(32),
    Tanh(),

    Linear(32, vocab_size)
])

emb_layer = Embedding(vocab_size, 10)

# Load target matrix weights explicitly using runtime asset paths
emb_layer.weight = np.load(get_asset_path("C.npy"))

for i, layer in enumerate(model.layers):
    if i == 1:
        layer.weight = np.load(get_asset_path("layer_1_weight.npy"))
    if i == 2:
        layer.gamma = np.load(get_asset_path("layer_2_gamma.npy"))
        layer.beta = np.load(get_asset_path("layer_2_beta.npy"))
        layer.running_mean = np.load(get_asset_path("layer_2_mean.npy"))
        layer.running_var = np.load(get_asset_path("layer_2_var.npy"))
    if i == 5:
        layer.weight = np.load(get_asset_path("layer_5_weight.npy"))
    if i == 6:
        layer.gamma = np.load(get_asset_path("layer_6_gamma.npy"))
        layer.beta = np.load(get_asset_path("layer_6_beta.npy"))
        layer.running_mean = np.load(get_asset_path("layer_6_mean.npy"))
        layer.running_var = np.load(get_asset_path("layer_6_var.npy"))
    if i == 9:
        layer.weight = np.load(get_asset_path("layer_9_weight.npy"))
    if i == 10:
        layer.gamma = np.load(get_asset_path("layer_10_gamma.npy"))
        layer.beta = np.load(get_asset_path("layer_10_beta.npy"))
        layer.running_mean = np.load(get_asset_path("layer_10_mean.npy"))
        layer.running_var = np.load(get_asset_path("layer_10_var.npy"))
    if i == 12:
        layer.weight = np.load(get_asset_path("layer_12_weight.npy"))
        layer.bias = np.load(get_asset_path("layer_12_bias.npy"))


# ------------------------------------------------------------------------------
# Inference Loop Logic
# ------------------------------------------------------------------------------
def generate_name(*args):
    """ Generates a single Pokemon name back out using the sequence framework """
    context = [0] * block_size
    out = []
    while True:
        # Wrap context array to form shape (1, 8) -> Embedding yields (1, 8, 10)
        emb = emb_layer(np.array([context]))
        logits = model(emb)
        
        # Flatten the 2D output (1, vocab_size) down to 1D array for tracking probabilities
        logits = logits.flatten()
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        
        ix = np.random.choice(vocab_size, p=probs)
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    return "".join(out).capitalize()


# ------------------------------------------------------------------------------
# Gradio Blocks Layout Definition
# ------------------------------------------------------------------------------
with gr.Blocks(title="pokepy") as demo:
    gr.Markdown("# pokepy")
    gr.Markdown("A character-level pokemon name generator built from scratch using NumPy.")
    
    with gr.Row():
        with gr.Column():
            generate_btn = gr.Button("Generate Name", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Generated Pokémon Name", interactive=False)
            
    # Explicitly map the click event without passing unhashable structures
    generate_btn.click(
        fn=generate_name,
        inputs=[],
        outputs=[output_text]
    )


def main():
    # 127.0.0.1 directly circumvents internal proxy exceptions on local developer setups
    print("Initializing PokePy local production instance...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)

if __name__ == "__main__":
    main()