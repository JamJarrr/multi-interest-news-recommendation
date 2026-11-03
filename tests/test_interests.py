import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.evaluation.controlled import select_cohort, intervene
from src.evaluation.detailed import evaluate_predictions
from src.evaluation.ranking import evaluate_ranking
from src.models.baselines import SharedProjection
from src.models.interests import build_model
from src.training.interest_data import index_impressions, sample_training
from src.training.train_b1 import build_training_arrays


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.h = torch.randn(3, 5, 384)
        self.mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
        self.c = torch.randn(3, 4, 384)

    def test_padding_and_empty_history(self):
        for name in ("b1", "b2", "b3_k2", "b3_k4", "b4"):
            m = build_model(name)
            actual = m(self.h, self.mask, self.c)
            changed = self.h.clone()
            changed[~self.mask] = 1000
            torch.testing.assert_close(actual, m(changed, self.mask, self.c))
            self.assertTrue(torch.isfinite(actual).all())
            interests, weights = m.encode_history(self.h, self.mask)
            self.assertEqual(float(weights[2].sum()), 0)
            F.cross_entropy(actual, torch.zeros(3, dtype=torch.long)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_b1_matches_existing_definition(self):
        old, new = SharedProjection(384), build_model("b1")
        new.projection.load_state_dict(old.projection.state_dict())
        means = (self.h * self.mask[:, :, None]).sum(1) / self.mask.sum(1).clamp_min(1)[:, None]
        torch.testing.assert_close(old(means, self.c), new(self.h, self.mask, self.c))

    def test_hard_max_matches_manual(self):
        m = build_model("b3_k4")
        interests, _ = m.encode_history(self.h, self.mask)
        candidate = F.normalize(m.projection(self.c), dim=-1)
        expected = (interests[:, :, None] * candidate[:, None]).sum(-1).max(1).values
        torch.testing.assert_close(m(self.h, self.mask, self.c), expected)

    def test_capacity_match(self):
        sizes = {n: sum(p.numel() for p in build_model(n).parameters())
                 for n in ("b1", "b2", "b3_k2", "b3_k4", "b4")}
        self.assertEqual(sizes, {"b1":24640, "b2":24704, "b3_k2":24768, "b3_k4":24896, "b4":24899})

    def test_checkpoint_roundtrip(self):
        m = build_model("b3_k4")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"weights.pt"
            torch.save(m.state_dict(), p)
            restored = build_model("b3_k4")
            restored.load_state_dict(torch.load(p, weights_only=True))
            torch.testing.assert_close(m(self.h, self.mask, self.c), restored(self.h, self.mask, self.c))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            ["1", "u1", "a a b b c c c c c c", "a-1 b-1 c-0 d-0"],
            ["2", "u1", "", "b-0 a-1"],
            ["3", "u2", "a a b b c c c c c c", "a-1 c-0 b-0"],
        ], columns=["impression_id","user_id","history","impressions"])
        self.index = dict(zip("abcd",range(4)))
        self.category = dict(zip("abcd","ABCD"))

    def test_negative_sampling_matches_b1(self):
        embeddings = np.random.default_rng(4).normal(size=(4,384)).astype(np.float32)
        _, expected = build_training_arrays(self.frame, embeddings, self.index, 50, 4, 42)
        rows, actual = sample_training(index_impressions(self.frame,self.index), 42, 4)
        np.testing.assert_array_equal(actual - 1, expected)
        self.assertEqual(len(rows), len(expected))

    def test_metrics_match_existing_evaluator(self):
        scores = [np.array([.2,.4,.3,.1]),np.array([.1,.2]),np.array([.3,.2,.1])]
        iterator = iter(scores)
        old = evaluate_ranking(self.frame, lambda h,c:next(iterator), self.category)
        new, detail = evaluate_predictions(self.frame,scores,self.category)
        for k,v in old.items():
            self.assertAlmostEqual(v,new[k])
        self.assertEqual(detail["impressions"].sum(),3)

    def test_controlled_ratios_are_paired_without_replacement(self):
        category = {f"a{i}":"A" for i in range(25)} | {f"b{i}":"B" for i in range(15)}
        history = " ".join(category)
        frame = pd.DataFrame([["1","u",history,"a1-1 b1-0"]],
                             columns=["impression_id","user_id","history","impressions"])
        cohort,designs=select_cohort(frame,category)
        for count in (10,16,18):
            changed, targets=intervene(cohort,designs,category,11,count)
            ids=changed.iloc[0].history.split()
            self.assertEqual(len(ids),20)
            self.assertEqual(len(set(ids)),20)
            self.assertEqual(sum(category[n]=="A" for n in ids),count)
            self.assertEqual(targets,["B"])
            self.assertEqual(changed.iloc[0].impressions,frame.iloc[0].impressions)
            self.assertTrue(set(ids)<=set(history.split()))
            again,_=intervene(cohort,designs,category,11,count)
            self.assertEqual(again.iloc[0].history,changed.iloc[0].history)



class EvaluationTests(unittest.TestCase):
    def test_user_macro_does_not_weight_active_users_more(self):
        category={"a":"A","b":"B"}
        candidates="a-1 "+" ".join(["b-0"]*11)
        frame=pd.DataFrame([["1","u1","a a b b b b b b b b",candidates],
                            ["2","u1","a a b b b b b b b b",candidates],
                            ["3","u2","a a b b b b b b b b",candidates]],
                           columns=["impression_id","user_id","history","impressions"])
        hit=np.array([2.]+[1.]*11); miss=np.array([0.]+[1.]*11)
        metrics,_=evaluate_predictions(frame,[hit,miss,hit],category)
        self.assertAlmostEqual(metrics["minority_recall_at_10_user_macro"],.75)
        self.assertAlmostEqual(metrics["recall_at_10"],2/3)

    def test_batched_predictor_matches_forward(self):
        from src.training.predict_interests import predict
        frame=pd.DataFrame([["1","u","a b","a-1 b-0"],["2","v","","b-1"]],
                           columns=["impression_id","user_id","history","impressions"])
        indexed=index_impressions(frame,{"a":0,"b":1},max_history=3)
        embeddings=torch.randn(3,384); embeddings[0]=0
        for name in ("b1","b2","b3_k2","b3_k4","b4"):
            model=build_model(name)
            predicted,_=predict(model,embeddings,indexed,torch.device("cpu"),1,True)
            h,c=indexed.tensors(torch.device("cpu"))
            with torch.no_grad():
                direct=model(embeddings[h],h!=0,embeddings[c]).numpy()
            for i,length in enumerate(indexed.lengths):
                np.testing.assert_allclose(predicted[i],direct[i,:length],atol=1e-6)


if __name__=="__main__":
    unittest.main()


