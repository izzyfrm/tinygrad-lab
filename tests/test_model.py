import unittest
import numpy as np
from tinygrad_lab import NeuralNetwork, make_moons


class NeuralNetworkTests(unittest.TestCase):
    def test_forward_shape_and_range(self):
        model = NeuralNetwork(seed=1)
        p = model.forward(np.zeros((7, 2)))
        self.assertEqual(p.shape, (7, 1))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_training_reduces_loss(self):
        x, y = make_moons(n=200, seed=4)
        model = NeuralNetwork(seed=4)
        initial = model.train_step(x, y, lr=0.1)
        for _ in range(100):
            last = model.train_step(x, y, lr=0.1)
        self.assertLess(last, initial)

    def test_reproducible_dataset(self):
        x1, y1 = make_moons(seed=3)
        x2, y2 = make_moons(seed=3)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


if __name__ == '__main__':
    unittest.main()
