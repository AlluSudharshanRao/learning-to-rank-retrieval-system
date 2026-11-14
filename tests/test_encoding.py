"""Verify pooling semantics on CPU; skip if optional PyTorch is unavailable."""

import importlib.util
import unittest

from ltr_system.encoding import masked_mean_pool


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class PoolingTests(unittest.TestCase):
    def test_padding_does_not_change_mean(self):
        import torch

        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [999.0, 999.0]]])
        result = masked_mean_pool(hidden, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(result, torch.tensor([[2.0, 4.0]]))

    def test_rejects_all_padding(self):
        import torch

        with self.assertRaises(ValueError):
            masked_mean_pool(torch.ones(1, 2, 3), torch.zeros(1, 2))

    def test_rejects_misaligned_mask(self):
        import torch

        with self.assertRaises(ValueError):
            masked_mean_pool(torch.ones(1, 2, 3), torch.ones(1, 3))
