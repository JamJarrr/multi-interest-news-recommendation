import unittest
import numpy as np
import torch
from experiments.followup_models import ScaledInterest, overlap_penalty, category_predictions
from src.models.interests import InterestRecommender
from src.training.interest_data import IndexedImpressions
from experiments.followup import choose, spec


class FollowupTests(unittest.TestCase):
    def test_scale_one_exact_original_and_gradient(self):
        torch.manual_seed(1)
        old=InterestRecommender(heads=4)
        new=ScaledInterest(4,1)
        new.load_state_dict(old.state_dict())
        x=torch.randn(3,6,384); candidates=torch.randn(3,5,384)
        mask=torch.tensor([[1,1,1,1,0,0],[0,0,0,0,0,0],[1,0,0,0,0,0]],dtype=torch.bool)
        torch.testing.assert_close(old(x,mask,candidates),new(x,mask,candidates),rtol=0,atol=0)
        scores,penalty=new.training_scores(x,mask,candidates)
        (scores.sum()+penalty).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in new.parameters()))

    def test_scale_is_inverse_temperature_and_padding(self):
        m=ScaledInterest(4,8)
        projected=torch.randn(2,5,64)
        mask=torch.tensor([[1,1,0,0,0],[0,0,0,0,0]],dtype=torch.bool)
        u,a=m.pool_projected(projected,mask)
        raw=torch.einsum("bld,kd->bkl",projected,m.queries)/8
        expected=torch.softmax(raw[0,:,:2]/.125,dim=-1)
        torch.testing.assert_close(a[0,:,:2],expected)
        self.assertEqual(a[:,:,2:].sum().item(),0)
        self.assertEqual(a[1].sum().item(),0)
        self.assertTrue(torch.isfinite(u).all())

    def test_diversity_penalty_extremes_and_short_histories(self):
        mask=torch.ones(1,4,dtype=torch.bool)
        uniform=torch.full((1,4,4),.25,requires_grad=True)
        separated=torch.eye(4)[None]
        self.assertAlmostEqual(overlap_penalty(uniform,mask).item(),1)
        self.assertAlmostEqual(overlap_penalty(separated,mask).item(),0)
        short=torch.tensor([[1,1,0,0]],dtype=torch.bool)
        loss=overlap_penalty(uniform,short)
        self.assertEqual(loss.item(),0)
        loss.backward()
        self.assertTrue(torch.isfinite(uniform.grad).all())

    def test_category_only_present_groups_and_empty_fallback(self):
        embeddings=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[-1.,0.]])
        indexed=IndexedImpressions([0,1,2],np.array([[1,2,0],[0,0,0],[1,0,0]]),
                                   np.array([[1,2,3]]*3),np.array([3]*3),[])
        cats=torch.tensor([-1,0,1,2])
        p=category_predictions(embeddings,indexed,cats,3)
        np.testing.assert_allclose(p[0],[1,1,0])
        np.testing.assert_allclose(p[1],[0,0,0])
        # Absent zero vectors must not cap a genuinely negative score at zero.
        np.testing.assert_allclose(p[2],[1,0,-1])

    def test_selection_respects_guardrail_then_minority(self):
        records=[{"spec":spec(4,a),"metrics":{"ndcg_at_10":n,"minority_recall_at_10_user_macro":r}}
                 for a,n,r in [(1,.4,.5),(8,.399,.6),(32,.3,.9)]]
        self.assertEqual(choose(records,.396)["spec"]["attention_scale"],8)
        self.assertEqual(choose(records,.5)["spec"]["attention_scale"],1)


if __name__=="__main__":
    unittest.main()