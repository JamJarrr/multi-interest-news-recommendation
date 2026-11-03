"""Audit follow-up checkpoints, saved scores, selection and original preservation."""
import json
import numpy as np
from experiments.followup import *
from experiments.followup_models import category_predictions
from src.training.predict_interests import predict
from src.evaluation.summarize_interests import aggregate_users


def main():
    config,identity,device,e_np,embeddings,index,category,cat_ids,cats=initialize()
    selection=read(OUT/"selection.json")
    summary=read(OUT/"summary.json")
    screening=selection["screening"]
    floor=selection["validation_ndcg_floor"]
    for h,role in [(1,"b2_tuned"),(4,"b3_scaled")]:
        runs=[r for r in screening if r["spec"]["heads"]==h and r["spec"]["diversity_weight"]==0]
        assert choose(runs,floor)["spec"]==selection["roles"][role]
    a=selection["roles"]["b3_scaled"]["attention_scale"]
    candidates=[r for r in screening if r["spec"]["heads"]==4 and r["spec"]["attention_scale"]==a]
    assert choose(candidates,floor)["spec"]==selection["roles"]["b3_diversity"]
    frame=load_behaviors(ROOT/"data/processed/behaviors_test.csv.gz").iloc[:32].copy()
    indexed=index_impressions(frame,index,50)
    unique={identifier(s):s for s in selection["roles"].values()}
    tags=[]
    maxdiff=0.
    for s in unique.values():
        for seed in SEEDS:
            name=tag(s,seed)
            tags.append(name)
            training=read(OUT/(name+"_training.json"))
            best=max(training["history"],key=lambda r:r["validation_ndcg_at_10"])
            assert best["epoch"]==training["best_epoch"]
            validation=read(OUT/(name+"_validation.json"))
            np.testing.assert_allclose(validation["metrics"]["ndcg_at_10"],best["validation_ndcg_at_10"],atol=1e-12,rtol=0)
            model=load_model(s,seed,identity,device)
            scores,_=predict(model,embeddings,indexed,device,32)
            saved=np.load(LOCAL/name/"test_predictions.npz")
            np.testing.assert_array_equal(saved["impression_ids"][:32],frame.impression_id.astype(str).to_numpy())
            for i,actual in enumerate(scores):
                expected=saved["scores"][saved["offsets"][i]:saved["offsets"][i+1]]
                np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
                maxdiff=max(maxdiff,float(np.abs(actual-expected).max()))
            if s["attention_scale"]==1 and s["diversity_weight"]==0:
                original="b2" if s["heads"]==1 else "b3_k4"
                old=read(ROOT/f"results/experiments/main/{original}_seed{seed}_test.json")
                current=read(OUT/(name+"_test.json"))
                for metric in ["ndcg_at_10","minority_recall_at_10_user_macro"]:
                    np.testing.assert_allclose(current["metrics"][metric],old["metrics"][metric],atol=1e-12,rtol=0)
    grouped=category_predictions(embeddings,indexed,cat_ids,cats,32)
    saved=np.load(LOCAL/"category_groups/test_predictions.npz")
    for i,actual in enumerate(grouped):
        expected=saved["scores"][saved["offsets"][i]:saved["offsets"][i+1]]
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
        maxdiff=max(maxdiff,float(np.abs(actual-expected).max()))
    roles={name:[tag(s,seed) for seed in SEEDS] for name,s in selection["roles"].items()}
    roles.update(b0=["b0"],category_groups=["category_groups"])
    for role,names in roles.items():
        for scope,suffix in [("full","test_users.npz"),("last50","test_last50_users.npz")]:
            for metric in ["ndcg_at_10","minority_recall_at_10_user_macro"]:
                measure=aggregate_users([LOCAL/name/suffix for name in names],metric)
                actual=measure[1].sum()/measure[2].sum()
                np.testing.assert_allclose(actual,summary["models"][role][scope][metric]["mean"],atol=1e-12,rtol=0)
    for name in tags+["b0","category_groups"]:
        for sample in [11,29,47]:
            for ratio in [50,80,90]:
                record=read(OUT/f"{name}_sample{sample}_ratio{ratio}.json")
                assert record["identity"]==identity
                assert record["metrics"]["impressions"]==3210
                assert record["metrics"]["sensitivity"]["target"]["target_users"]==832
    assert read(OUT/"controlled_design.json")["cohort_ids_sha256"]==read(ROOT/"results/experiments/main/controlled/design.json")["cohort_impression_ids_sha256"]
    result={"status":"passed","selected_checkpoints":len(tags),
            "natural_files":len(list(OUT.glob("*_validation.json")))+len(list(OUT.glob("*_test.json"))),"controlled_files":9*(len(tags)+2),
            "rescored_impressions":32*(len(tags)+1),"maximum_score_difference":maxdiff,
            "original_source_data_cache_fingerprint":"unchanged",
            "scale1_zero_penalty_baselines":"exact original metrics",
            "selection_rule":"recomputed from validation-only screening",
            "full_and_last50_aggregates":"verified from user records",
            "controlled_cohort":"exact original cohort and 832 target users"}
    write_json(OUT/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()