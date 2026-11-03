import unittest
import torch

from experiments.masked_history_models import masked_history_loss
from experiments.soft_clustering_models import SoftClusterRecommender


class MaskedHistoryTests(unittest.TestCase):
    def test_masked_loss_is_finite_and_differentiable(self):
        torch.manual_seed(4)
        model = SoftClusterRecommender(dimension=6, bottleneck=2, clusters=2)
        embeddings = torch.randn(20, 6)
        histories = torch.tensor([
            [1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0],
            [10, 11, 12, 0], [13, 14, 15, 0], [16, 17, 18, 0]])
        loss, count = masked_history_loss(model, embeddings, histories)
        self.assertEqual(count, 6)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_short_histories_return_zero(self):
        model = SoftClusterRecommender(dimension=3, bottleneck=2, clusters=2)
        embeddings = torch.randn(5, 3)
        histories = torch.tensor([[1, 0], [2, 0]])
        loss, count = masked_history_loss(model, embeddings, histories)
        self.assertEqual(count, 0)
        self.assertEqual(float(loss), 0.)


if __name__ == "__main__":
    unittest.main()
