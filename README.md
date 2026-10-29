# TinyGrad Lab

A compact, readable neural network **built from scratch in NumPy**. No PyTorch, pretrained models, API keys, or GPU required.

## Why this project?

This educational project implements the core mechanics of machine learning instead of hiding them behind a framework:

- Dense layers and He initialization
- ReLU and sigmoid activations
- Binary cross-entropy loss
- Manual backpropagation and gradient descent
- Train/test evaluation on a reproducible two-moons dataset
- Automated unit tests

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python train.py
```

You'll see training loss at intervals and accuracy on a held-out test set. Results vary with training settings and random seeds.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Experiments

```bash
python train.py --epochs 2000 --lr 0.05 --seed 123
```

Try changing the hidden layer size in `train.py`, the learning rate, and the dataset noise. Keep the test set untouched when comparing models.

## Repo layout

```text
src/tinygrad_lab/model.py  # Forward pass, gradients and optimizer step
src/tinygrad_lab/data.py   # Synthetic dataset
train.py                   # Training loop and evaluation
tests/                     # Automated tests
```

## Next research directions

1. Compare activation functions and measure convergence.
2. Add minibatch stochastic gradient descent.
3. Implement gradient checking with finite differences.
4. Benchmark against an equivalent PyTorch model.
5. Publish a short report with metrics across 5 seeds.

## License

MIT
