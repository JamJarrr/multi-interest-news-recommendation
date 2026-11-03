import unittest
import numpy as np
from src.evaluation.summarize_interests import bootstrap_measures, user_measure


class StatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_uses_identical_user_draws(self):
        users=np.array(["u1","u2","u3"])
        counts=np.array([1.,7.,2.])
        base=np.array([.2,2.1,1.6])
        result=bootstrap_measures({"a":(users,base,counts),
                                   "b":(users,base+.1*counts,counts)},samples=80,seed=12)
        np.testing.assert_allclose(result["b"]-result["a"],.1,atol=1e-12)
        repeated=bootstrap_measures({"a":(users,base,counts)},samples=80,seed=12)
        np.testing.assert_array_equal(result["a"],repeated["a"])

    def test_user_macro_denominator_is_eligible_users(self):
        detail={"hits_20":np.array([1,1,0]),"targets_20":np.array([2,10,0])}
        values,counts=user_measure(detail,"minority_recall_at_10_user_macro")
        self.assertAlmostEqual(values.sum()/counts.sum(),.3)

    def test_bootstrap_rejects_different_cohorts(self):
        with self.assertRaises(ValueError):
            bootstrap_measures({"a":(np.array(["u"]),np.ones(1),np.ones(1)),
                                "b":(np.array(["v"]),np.ones(1),np.ones(1))},samples=2)

if __name__=="__main__":
    unittest.main()

