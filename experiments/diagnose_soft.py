"""Read-only diagnostic separating soft pooling from adapter training."""
import json
from pathlib import Path

from experiments.soft_clustering_models import SoftClusterRecommender, predict_soft
from src.evaluation.detailed import evaluate_predictions
from src.training.common import load_behaviors
from src.training.interest_data import index_impressions
from src.training.run_interests import prepare


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/main.json").read_text())
    device, _, embeddings, index, _, category = prepare(root, config)
    frame = load_behaviors(root / "data/processed/behaviors_validation.csv.gz")
    indexed = index_impressions(frame, index, 50)
    model = SoftClusterRecommender().to(device)
    predictions, diagnostics = predict_soft(model, embeddings, indexed, 256)
    metrics, _ = evaluate_predictions(frame, predictions, category, (.1, .2, .3))
    print(json.dumps({"untrained_soft_ndcg10": metrics["ndcg_at_10"],
                      "untrained_soft_minority": metrics["minority_recall_at_10_user_macro"],
                      "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
