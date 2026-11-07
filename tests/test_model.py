import unittest

import torch

from edge_learning.model import EdgeTextureCNN


class EdgeTextureCNNTests(unittest.TestCase):
    def test_forward_shapes(self):
        model = EdgeTextureCNN().eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            logits, edges, reconstructed = model(images)
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertEqual(tuple(edges.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(reconstructed.shape), (2, 1, 32, 32))

    def test_backward_has_gradients(self):
        model = EdgeTextureCNN().train()
        images = torch.randn(2, 3, 32, 32)
        logits, edges, reconstructed = model(images)
        loss = logits.square().mean() + (edges - reconstructed).abs().mean()
        loss.backward()
        self.assertIsNotNone(model.texture1.layers[0].weight.grad)
        self.assertIsNotNone(model.shape1.layers[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
