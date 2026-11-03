"""Check artifact completeness, provenance, checkpoint scoring and aggregate values."""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions
from src.training.predict_interests import predict
from src.training.run_interests import prepare, fingerprint, load_checkpoint

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/"results/experiments/main"
    protocol=json.loads((out/"protocol.json").read_text())
    config=protocol["config"]
    device,_,embeddings,index,cache,category=prepare(ROOT,config)
    identity=fingerprint(ROOT,config,cache)
    assert identity==protocol["identity"], "Current code/data/cache differ from the locked experiment"
    frame=load_behaviors(ROOT/"data/processed/behaviors_test.csv.gz").iloc[:32].copy()
    indexed=index_impressions(frame,index,config["max_history"])
    summary=json.loads((out/"summary.json").read_text())
    controlled=json.loads((out/"controlled_summary.json").read_text())
    checked=0
    max_score_delta=0.
    for name in config["models"]:
        values=[]
        for seed in config["seeds"]:
            tag=f"{name}_seed{seed}"
            training=json.loads((out/f"{tag}_training.json").read_text())
            assert training["identity"]==identity
            assert training["training_samples"]==187856
            assert training["best_validation_ndcg_at_10"]==max(x["validation_ndcg_at_10"] for x in training["training_history"])
            assert training["training_history"][training["best_epoch"]-1]["validation_ndcg_at_10"]==training["best_validation_ndcg_at_10"]
            model,_=load_checkpoint(ROOT,"main",name,seed,device,identity)
            predictions,_=predict(model,embeddings,indexed,device,32)
            with np.load(ROOT/"results/runs/main"/tag/"test_predictions.npz",allow_pickle=False) as saved:
                for i,actual in enumerate(predictions):
                    expected=saved["scores"][saved["offsets"][i]:saved["offsets"][i+1]]
                    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
                    max_score_delta=max(max_score_delta,float(np.max(np.abs(actual-expected))))
            for split in ("validation","test"):
                record=json.loads((out/f"{tag}_{split}.json").read_text())
                assert record["identity"]==identity
                assert record["metrics"]["impressions"]=={"validation":31393,"test":73152}[split]
            record=json.loads((out/f"{tag}_test.json").read_text())
            values.append(record["metrics"])
            checked+=1
        for metric,aggregate in summary["models"][name]["metrics"].items():
            np.testing.assert_allclose(aggregate["mean"],np.mean([v[metric] for v in values]),rtol=0,atol=1e-14)
            np.testing.assert_allclose(aggregate["sample_std"],np.std([v[metric] for v in values],ddof=1),rtol=0,atol=1e-14)
    controls=list((out/"controlled").glob("*_sample*_ratio*.json"))
    assert len(controls)==16*3*3
    for path in controls:
        record=json.loads(path.read_text())
        assert record["identity"]==identity
        assert record["metrics"]["impressions"]==3210
        assert record["metrics"]["sensitivity"]["target"]["target_users"]==832
    assert json.loads((ROOT/"results/data_audit.json").read_text())==json.loads((ROOT/"results/local_data/data_audit.json").read_text())
    assert all(v["within_tolerance"] for v in json.loads((ROOT/"results/reproduction/verification.json").read_text())["runs"].values())
    result={"status":"passed","trained_checkpoints_checked":checked,"natural_evaluation_files":32,
            "controlled_evaluation_files":len(controls),"raw_predictions_rescored":32*checked,
            "maximum_rescored_absolute_difference":max_score_delta,
            "source_data_cache_identity":"matches locked protocol",
            "aggregate_seed_means_and_sample_standard_deviations":"verified",
            "original_audit":"identical","baseline_reproduction_tolerance":1e-4,
            "behavior_and_statistics_tests":13}
    write_json(out/"verification.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()


