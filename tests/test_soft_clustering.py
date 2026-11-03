import unittest
import torch

from experiments.soft_clustering_models import SoftClusterRecommender


class SoftClusteringTests(unittest.TestCase):
    def test_zero_adapter_preserves_normalized_input(self):
        model = SoftClusterRecommender(dimension=4, bottleneck=2)
        values = torch.randn(3, 4)
        torch.testing.assert_close(model.adapt(values), torch.nn.functional.normalize(values, dim=-1))

    def test_competitive_assignments_and_gradients(self):
        torch.manual_seed(3)
        model = SoftClusterRecommender(dimension=4, bottleneck=2, clusters=2)
        history = torch.randn(3, 5, 4)
        mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                            dtype=torch.bool)
        candidates = torch.randn(3, 4, 4)
        scores, compactness = model(history, mask, candidates)
        _, active, weights, _ = model.encode_history(history, mask)
        torch.testing.assert_close(weights.sum(-1)[mask], torch.ones(int(mask.sum())))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.isfinite(compactness))
        self.assertFalse(active[2].any())
        (scores.sum() + compactness).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
