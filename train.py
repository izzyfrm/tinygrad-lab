"""Train and evaluate our NumPy neural network, without APIs or a GPU."""
import argparse
import numpy as np
from tinygrad_lab import NeuralNetwork, make_moons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    x, y = make_moons(n=800, seed=args.seed)
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    model = NeuralNetwork(seed=args.seed)
    for epoch in range(1, args.epochs + 1):
        loss = model.train_step(x_train, y_train, lr=args.lr)
        if epoch == 1 or epoch % 200 == 0:
            print(f'Epoch {epoch:4d} | train loss: {loss:.4f}')
    train_accuracy = np.mean(model.predict(x_train) == y_train)
    test_accuracy = np.mean(model.predict(x_test) == y_test)
    print(f'Train accuracy: {train_accuracy:.1%}')
    print(f'Test accuracy:  {test_accuracy:.1%}')


if __name__ == '__main__':
    main()
