"""Exploratory follow-up: validation-only selection, then locked test evaluation."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from experiments.followup_models import ScaledInterest, category_predictions, attention_diagnostics
from src.training.run_interests import prepare, fingerprint, seed_everything, save_predictions
from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions, sample_training
from src.training.predict_interests import predict, predict_b0
from src.evaluation.detailed import evaluate_predictions
from src.evaluation.ranking import _ndcg
from src.evaluation.controlled import select_cohort, intervene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/followup"
LOCAL = ROOT / "results/runs/followup"
WEIGHTS = ROOT / "checkpoints/followup"
SEEDS = [42, 2026, 3407]
SCALES = [1, 8, 32]
LAMBDAS = [0, .01, .1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def spec(heads, scale, penalty=0):
    return dict(heads=int(heads), attention_scale=float(scale), diversity_weight=float(penalty))


def identifier(s):
    return f'k{s["heads"]}_a{s["attention_scale"]:g}_d{s["diversity_weight"]:g}'


def tag(s, seed):
    return identifier(s) + f"_seed{seed}"


def initialize():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    original = read(ROOT / "results/experiments/main/protocol.json")
    config = original["config"]
    device, e_np, embeddings, index, cache, category = prepare(ROOT, config)
    base = fingerprint(ROOT, config, cache)
    if base != original["identity"]:
        raise ValueError("Original experiment identity has changed")
    identity = {"base": base, "followup_sources": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "experiments/followup_models.py",
                  ROOT / "src/evaluation/controlled.py"]}}
    protocol = dict(
        identity=identity, seeds=SEEDS, screening_seed=42,
        attention_scales=SCALES, diversity_weights=LAMBDAS,
        training="Original optimizer, negative sampling, 20 epochs / patience 3; best validation NDCG@10 checkpoint",
        selection="Among configs within 1% relative validation NDCG@10 of original B2 seed42, maximize full-history minority Recall@10; ties: NDCG then smaller scale/penalty. If none, maximize NDCG then minority recall.",
        sequence="Screen B2 and B3 K4 at scales 1,8,32. Fix best B3 scale; screen diversity 0,.01,.1. Lock selected configs before new test evaluation.",
        category_control="Frozen normalized MiniLM; one mean vector for each present history category; max cosine; only last50 history; no candidate category or label in score. Extra metadata and variable K, diagnostic not capacity matched.",
        evaluation="Full-history and last50 minority definitions; natural and fixed-cohort 50:50/80:20/90:10; three sampling seeds; user paired bootstrap conditional on fitted models.",
        exploration="Earlier test outcomes informed this follow-up. This is exploratory, not a fresh independent confirmation.",
        decision="Ranking-loss temperature stays 1; no simultaneous temperature/lr sweep. First isolate attention scale, head count and diversity.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        saved = read(path)
        if {k:v for k,v in saved.items() if k != "created_at_utc"} != protocol:
            raise ValueError("Follow-up protocol differs; use a separate experiment")
    else:
        write_json(path, {**protocol, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    cats = sorted(set(category.values()))
    cat_index = {c:i for i,c in enumerate(cats)}
    cat_ids = np.full(len(index)+1, -1, np.int64)
    for news, i in index.items():
        cat_ids[i+1] = cat_index[category[news]]
    return config, identity, device, e_np, embeddings, index, category, torch.as_tensor(cat_ids, device=device), len(cats)


def load_model(s, seed, identity, device):
    record = torch.load(WEIGHTS / (tag(s,seed)+".pt"), weights_only=True, map_location="cpu")
    if record["identity"] != identity or record["spec"] != s:
        raise ValueError("Checkpoint identity mismatch")
    model = ScaledInterest(s["heads"], s["attention_scale"]).to(device)
    model.load_state_dict(record["state_dict"])
    return model


def train(s, seed, config, identity, embeddings, training, validation, device):
    name = tag(s, seed)
    path = OUT / (name + "_training.json")
    if path.exists():
        load_model(s, seed, identity, device)
        return read(path)
    if s["attention_scale"] == 1 and s["diversity_weight"] == 0:
        original_name = "b2" if s["heads"] == 1 else "b3_k4"
        old = torch.load(ROOT / f"checkpoints/main/{original_name}_seed{seed}.pt",
                         weights_only=True, map_location="cpu")
        if old["identity"] != identity["base"]:
            raise ValueError("Original checkpoint mismatch")
        record = read(ROOT / f"results/experiments/main/{original_name}_seed{seed}_training.json")
        payload = {"spec":s, "seed":seed, "identity":identity, "origin":f"main/{original_name}",
                   "history":record["training_history"], "best_epoch":record["best_epoch"],
                   "parameters":record["trainable_parameters"], "seconds":0}
        torch.save({"spec":s, "seed":seed, "identity":identity, "state_dict":old["state_dict"]},
                   WEIGHTS / (name+".pt"))
        write_json(path,payload)
        return payload
    seed_everything(seed)
    rows, groups = sample_training(training, seed, config["negatives"])
    histories = torch.as_tensor(training.histories, device=device, dtype=torch.long)
    sample_rows = torch.as_tensor(rows, device=device)
    candidate_groups = torch.as_tensor(groups, device=device)
    model = ScaledInterest(s["heads"], s["attention_scale"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed)
    best, stale, best_epoch, state = -1., 0, 0, None
    records=[]
    started=time.perf_counter()
    for epoch in range(1, config["epochs"]+1):
        model.train()
        order=rng.permutation(len(rows))
        ce_total=penalty_total=0.
        for start in range(0,len(order),config["batch_size"]):
            batch=torch.as_tensor(order[start:start+config["batch_size"]], device=device)
            h=histories[sample_rows[batch]]
            scores, penalty=model.training_scores(embeddings[h],h!=0,embeddings[candidate_groups[batch]])
            ce=F.cross_entropy(scores,torch.zeros(len(batch),dtype=torch.long,device=device))
            loss=ce+s["diversity_weight"]*penalty
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ce_total+=ce.detach().item()*len(batch)
            penalty_total+=penalty.detach().item()*len(batch)
        predictions,_=predict(model,embeddings,validation,device,config["eval_batch_size"])
        ndcg=float(np.mean([_ndcg(y,np.argsort(-v,kind="stable"),10)
                            for y,v in zip(validation.labels,predictions,strict=True)]))
        rec={"epoch":epoch,"train_loss":ce_total/len(rows),
             "overlap_penalty":penalty_total/len(rows),"validation_ndcg_at_10":ndcg}
        records.append(rec)
        print(json.dumps({"training":name,**rec}),flush=True)
        if ndcg>best:
            best,best_epoch,stale=ndcg,epoch,0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
        if stale>=config["patience"]:
            break
    torch.save({"spec":s,"seed":seed,"identity":identity,"state_dict":state},WEIGHTS/(name+".pt"))
    payload={"spec":s,"seed":seed,"identity":identity,"origin":"followup",
             "history":records,"best_epoch":best_epoch,
             "parameters":sum(p.numel() for p in model.parameters()),
             "seconds":time.perf_counter()-started}
    write_json(path,payload)
    return payload


def score(name, model, frame, indexed, config, identity, e_np, embeddings, category,
          cat_ids, cats, device, split, targets=None, diagnostics=True):
    path=OUT / (name+"_"+split+".json")
    local=LOCAL/name
    local.mkdir(parents=True,exist_ok=True)
    if path.exists():
        saved=read(path)
        if saved["identity"]!=identity:
            raise ValueError("Score identity mismatch")
        if not (local/(split+"_users.npz")).exists():
            raise ValueError("Incomplete scores")
        return saved
    if name=="category_groups":
        predictions=category_predictions(embeddings,indexed,cat_ids,cats)
        diagnostic={}
    elif name=="b0":
        predictions=predict_b0(e_np,indexed)
        diagnostic={}
    else:
        predictions,diagnostic=predict(model,embeddings,indexed,device,256,diagnostics=diagnostics)
        if diagnostics:
            diagnostic.update(attention_diagnostics(model,embeddings,indexed))
    metrics,detail=evaluate_predictions(frame,predictions,category,(.1,.2,.3),targets)
    np.savez_compressed(local/(split+"_users.npz"),**detail)
    save_predictions(local/(split+"_predictions.npz"),frame,predictions)
    truncated=frame.copy()
    truncated["history"]=truncated.history.fillna("").map(lambda h:" ".join(h.split()[-50:]))
    last50,last_detail=evaluate_predictions(truncated,predictions,category,(.1,.2,.3),targets)
    np.savez_compressed(local/(split+"_last50_users.npz"),**last_detail)
    result={"name":name,"split":split,"identity":identity,
            "metrics":metrics,"last50_metrics":last50,"diagnostics":diagnostic}
    write_json(path,result)
    print(json.dumps({"evaluated":name,"split":split,"ndcg":metrics["ndcg_at_10"],
                      "minority":metrics["minority_recall_at_10_user_macro"]}),flush=True)
    return result


def choose(candidates, floor):
    eligible=[r for r in candidates if r["metrics"]["ndcg_at_10"]>=floor]
    if eligible:
        return max(eligible,key=lambda r:(r["metrics"]["minority_recall_at_10_user_macro"],
                                          r["metrics"]["ndcg_at_10"],
                                          -r["spec"]["attention_scale"],-r["spec"]["diversity_weight"]))
    return max(candidates,key=lambda r:(r["metrics"]["ndcg_at_10"],
                                        r["metrics"]["minority_recall_at_10_user_macro"],
                                        -r["spec"]["attention_scale"],-r["spec"]["diversity_weight"]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["select","evaluate","controlled"])
    args=parser.parse_args()
    config,identity,device,e_np,embeddings,index,category,cat_ids,cats=initialize()
    def evaluate_spec(s,seed,frame,indexed,split,targets=None,diagnostics=True):
        model=load_model(s,seed,identity,device)
        record=score(tag(s,seed),model,frame,indexed,config,identity,e_np,embeddings,category,
                     cat_ids,cats,device,split,targets,diagnostics)
        return {**record,"spec":s}
    selection_path=OUT/"selection.json"
    if args.stage=="select":
        train_frame=load_behaviors(ROOT/"data/processed/behaviors_train.csv.gz")
        val_frame=load_behaviors(ROOT/"data/processed/behaviors_validation.csv.gz")
        training=index_impressions(train_frame,index,50)
        validation=index_impressions(val_frame,index,50)
        def screen(s):
            train(s,42,config,identity,embeddings,training,validation,device)
            return evaluate_spec(s,42,val_frame,validation,"validation")
        if selection_path.exists():
            selection=read(selection_path)
            if selection["identity"]!=identity:
                raise ValueError("Selection identity mismatch")
        else:
            screened={h:[screen(spec(h,a)) for a in SCALES] for h in [1,4]}
            floor=.99*screened[1][0]["metrics"]["ndcg_at_10"]
            best_b2=choose(screened[1],floor)
            best_b3=choose(screened[4],floor)
            penalty_runs=[best_b3]+[screen(spec(4,best_b3["spec"]["attention_scale"],w)) for w in LAMBDAS[1:]]
            best_div=choose(penalty_runs,floor)
            roles={"b2_original":spec(1,1),"b3_original":spec(4,1),
                   "b2_tuned":best_b2["spec"],"b2_matched":spec(1,best_b3["spec"]["attention_scale"]),
                   "b3_scaled":best_b3["spec"],"b3_diversity":best_div["spec"]}
            # Keep strongest predeclared penalty as mechanism diagnostic regardless of selection.
            roles["b3_diversity_diagnostic"]=spec(4,best_b3["spec"]["attention_scale"],.1)
            selection={"identity":identity,"locked_at_utc":datetime.now(timezone.utc).isoformat(),
                       "validation_ndcg_floor":floor,"roles":roles,
                       "screening":[{"spec":r["spec"],"metrics":r["metrics"],"diagnostics":r["diagnostics"]}
                                    for r in screened[1]+screened[4]+penalty_runs[1:]],
                       "diagnostic_note":"lambda=.1 retained a priori for mechanism analysis, not claimed as validation winner"}
            write_json(selection_path,selection)
            print(json.dumps({"locked_roles":roles}),flush=True)
        unique={identifier(s):s for s in selection["roles"].values()}
        for s in unique.values():
            for seed in SEEDS:
                train(s,seed,config,identity,embeddings,training,validation,device)
                evaluate_spec(s,seed,val_frame,validation,"validation")
        for name in ["b0","category_groups"]:
            score(name,None,val_frame,validation,config,identity,e_np,embeddings,category,cat_ids,cats,device,"validation")
        write_json(OUT/"training_complete.json",{"identity":identity,"specifications":list(unique.values()),
                                                "seeds":SEEDS,"completed_at_utc":datetime.now(timezone.utc).isoformat()})
    else:
        selection=read(selection_path)
        completion=read(OUT/"training_complete.json")
        if selection["identity"]!=identity or completion["identity"]!=identity:
            raise ValueError("Must lock selection and complete training before test evaluation")
        unique={identifier(s):s for s in selection["roles"].values()}
        for s in unique.values():
            for seed in SEEDS:
                load_model(s,seed,identity,device)
        frame=load_behaviors(ROOT/"data/processed/behaviors_test.csv.gz")
        if args.stage=="evaluate":
            indexed=index_impressions(frame,index,50)
            for s in unique.values():
                for seed in SEEDS:
                    evaluate_spec(s,seed,frame,indexed,"test")
            for name in ["b0","category_groups"]:
                score(name,None,frame,indexed,config,identity,e_np,embeddings,category,cat_ids,cats,device,"test")
        else:
            cohort,designs=select_cohort(frame,category)
            write_json(OUT/"controlled_design.json",{"impressions":len(cohort),"users":int(cohort.user_id.nunique()),
                "fraction_of_test":len(cohort)/len(frame),"sampling_seeds":[11,29,47],
                "cohort_ids_sha256":hashlib.sha256("\n".join(cohort.impression_id).encode()).hexdigest()})
            for sample in [11,29,47]:
                for dominant in [10,16,18]:
                    changed,targets=intervene(cohort,designs,category,sample,dominant)
                    indexed=index_impressions(changed,index,50)
                    split=f"sample{sample}_ratio{dominant*5}"
                    for s in unique.values():
                        for seed in SEEDS:
                            evaluate_spec(s,seed,changed,indexed,split,targets,False)
                    for name in ["b0","category_groups"]:
                        score(name,None,changed,indexed,config,identity,e_np,embeddings,category,cat_ids,cats,device,split,targets,False)


if __name__=="__main__":
    main()