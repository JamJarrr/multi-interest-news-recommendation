import unittest
import numpy as np
import torch

from experiments.clustering_models import spherical_kmeans, random_partition


class ClusteringTests(unittest.TestCase):
    def test_spherical_kmeans_separates_two_directions_and_padding(self):
        items = torch.tensor([[[1., 0.], [.9, .1], [-1., 0.], [-.9, -.1], [0., 0.]]])
        mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
        centers, active, assignment = spherical_kmeans(items, mask, 2)
        self.assertTrue(active.all())
        self.assertEqual(assignment[0, 4], -1)
        self.assertEqual(assignment[0, 0], assignment[0, 1])
        self.assertEqual(assignment[0, 2], assignment[0, 3])
        self.assertNotEqual(assignment[0, 0], assignment[0, 2])
        self.assertLess(float(centers[0, 0] @ centers[0, 1]), -.9)

    def test_short_history_has_inactive_slots(self):
        items = torch.tensor([[[1., 0.], [0., 0.], [0., 0.]]])
        mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)
        _, active, assignment = spherical_kmeans(items, mask, 4)
        np.testing.assert_array_equal(active.numpy(), [[True, False, False, False]])
        np.testing.assert_array_equal(assignment.numpy(), [[0, -1, -1]])

    def test_random_partition_is_reproducible_and_uses_slots(self):
        items = torch.randn(2, 6, 3)
        mask = torch.tensor([[1] * 6, [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        out1 = random_partition(items, mask, 4, torch.Generator().manual_seed(7))
        out2 = random_partition(items, mask, 4, torch.Generator().manual_seed(7))
        for left, right in zip(out1, out2):
            torch.testing.assert_close(left, right)
        self.assertEqual(len(torch.unique(out1[2][0])), 4)
        np.testing.assert_array_equal(out1[1][1].numpy(), [True, True, False, False])


if __name__ == "__main__":
    unittest.main()
