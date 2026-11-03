"""Seed summaries, paired user bootstrap, and publication-style figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.training.common import write_json

METRICS = ("ndcg_at_5", "ndcg_at_10", "recall_at_10", "minority_recall_at_10_user_macro")


def read(path):
    return json.loads(path.read_text())


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)) if len(values)>1 else 0.,
            "values": values.tolist()}


def user_measure(detail, metric):
    if metric in ("ndcg_at_5", "ndcg_at_10", "recall_at_10"):
        return detail[metric + "_sum"].astype(float), detail["impressions"].astype(float)
    key = "target" if metric == "target_category_recall_at_10_user_macro" else "20"
    denominator = detail["targets_" + key]
    valid = denominator > 0
    ratio = np.divide(detail["hits_" + key], denominator, out=np.zeros(len(valid), float), where=valid)
    return ratio, valid.astype(float)


def aggregate_users(paths, metric):
    records = [dict(np.load(p, allow_pickle=False)) for p in paths]
    ids = records[0]["user_ids"]
    pairs = []
    for record in records:
        if not np.array_equal(ids, record["user_ids"]):
            raise ValueError("User cohorts do not align")
        pairs.append(user_measure(record, metric))
    denominators = [p[1] for p in pairs]
    if not all(np.array_equal(denominators[0], d) for d in denominators):
        raise ValueError("Metric denominator changes between runs")
    return ids, np.mean([p[0] for p in pairs], axis=0), denominators[0]


def bootstrap_measures(measures, samples=1000, seed=8675309):
    names = list(measures)
    size = len(measures[names[0]][0])
    if not size:
        raise ValueError("No users")
    for record in measures.values():
        if not np.array_equal(record[0], measures[names[0]][0]):
            raise ValueError("Paired comparison needs the same user IDs")
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(samples) for name in names}
    for start in range(0, samples, 20):
        count = min(20, samples-start)
        index = rng.integers(0, size, size=(count, size))
        for name, (_, numerator, denominator) in measures.items():
            den = denominator[index].sum(axis=1)
            if np.any(den == 0):
                raise ValueError("Bootstrap replicate contains no eligible users")
            draws[name][start:start+count] = numerator[index].sum(axis=1) / den
    return draws


def ci(values):
    return [float(x) for x in np.quantile(values, [.025,.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run", default="main")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "results/experiments" / args.run
    config = read(out / "protocol.json")["config"]
    variants = ["b0"] + config["models"]
    seeds = config["seeds"]
    summary = {"protocol": str((out/"protocol.json").relative_to(root)), "models": {},
               "bootstrap": {"unit":"user", "samples":config["bootstrap_samples"],
                             "seed":config["bootstrap_seed"],
                             "interpretation":"95% percentile intervals conditional on fitted models; predictions averaged over training seeds before user resampling. Seed SD is reported separately. Paired differences use identical user draws."}}
    paths = {}
    for name in variants:
        tags = [name] if name == "b0" else [f"{name}_seed{s}" for s in seeds]
        runs = [read(out/f"{tag}_test.json") for tag in tags]
        paths[name] = [root/"results/runs"/args.run/tag/"test_users.npz" for tag in tags]
        training = [] if name=="b0" else [read(out/f"{tag}_training.json") for tag in tags]
        summary["models"][name] = {
            "metrics": {metric:stats([r["metrics"][metric] for r in runs]) for metric in METRICS},
            "top10_minority_share":stats([r["metrics"]["top10_minority_share"] for r in runs]),
            "trainable_parameters": training[0]["trainable_parameters"] if training else 0,
            "training_and_validation_seconds":stats([r["training_and_validation_seconds"] for r in training]) if training else None,
            "peak_cuda_allocated_mb":max(r["peak_cuda_allocated_mb"] or 0 for r in training) if training else 0,
            "best_epochs":[r["best_epoch"] for r in training],
            "attention_diagnostics":[r["attention_diagnostics"] for r in runs],
            "coverage":{k:runs[0]["metrics"][k] for k in ("impressions","unique_users","minority_target_users","minority_positive_targets","minority_impressions")},
            "threshold_sensitivity":{key:stats([r["metrics"]["sensitivity"][key]["recall_at_10_user_macro"] for r in runs])
                                     for key in ("10","20","30")},
        }
    differences = {}
    for metric in METRICS:
        measures = {name:aggregate_users(paths[name],metric) for name in variants}
        draws = bootstrap_measures(measures,config["bootstrap_samples"],config["bootstrap_seed"])
        for name in variants:
            summary["models"][name]["metrics"][metric]["user_bootstrap_95_ci"] = ci(draws[name])
        for baseline in ("b0","b1","b2","b4"):
            for name in ("b3_k2","b3_k4"):
                key = f"{name}_minus_{baseline}"
                differences.setdefault(key,{})[metric] = {
                    "mean_difference":summary["models"][name]["metrics"][metric]["mean"]-summary["models"][baseline]["metrics"][metric]["mean"],
                    "paired_user_bootstrap_95_ci":ci(draws[name]-draws[baseline])}
    summary["paired_differences"] = differences
    write_json(out/"summary.json",summary)
    with (out/"model_comparison.csv").open("w",newline="") as f:
        fields=["model","parameters"]+[m+s for m in METRICS for s in ("_mean","_seed_std","_ci_low","_ci_high")]
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for name,record in summary["models"].items():
            row={"model":name,"parameters":record["trainable_parameters"]}
            for metric,value in record["metrics"].items():
                row.update({metric+"_mean":value["mean"],metric+"_seed_std":value["sample_std"],
                            metric+"_ci_low":value["user_bootstrap_95_ci"][0],metric+"_ci_high":value["user_bootstrap_95_ci"][1]})
            writer.writerow(row)
    controlled = {"design":read(out/"controlled/design.json"),"models":{}}
    metric="target_category_recall_at_10_user_macro"
    for name in variants:
        tags=[name] if name=="b0" else [f"{name}_seed{s}" for s in seeds]
        controlled["models"][name]={}
        for ratio in (50,80,90):
            training_means=[]
            users=[]
            by_training_seed={}
            for tag in tags:
                values=[]
                for sample in config["controlled"]["sampling_seeds"]:
                    record=read(out/"controlled"/f"{tag}_sample{sample}_ratio{ratio}.json")
                    values.append(record["metrics"][metric])
                    users.append(root/"results/runs"/args.run/"controlled"/tag/f"sample{sample}_ratio{ratio}_users.npz")
                training_means.append(float(np.mean(values)))
                by_training_seed[tag]={"sampling_results":values,"sampling_mean":float(np.mean(values))}
            measure=aggregate_users(users,metric)
            draws=bootstrap_measures({name:measure},config["bootstrap_samples"],config["bootstrap_seed"])[name]
            controlled["models"][name][str(ratio)]={**stats(training_means),"by_training_seed":by_training_seed,
                                                   "user_bootstrap_95_ci":ci(draws),
                                                   "eligible_target_users":int(measure[2].sum())}
    # Pair models on the same users, averaging over the same history sampling seeds.
    controlled["paired_differences"]={}
    for ratio in (50,80,90):
        measures={}
        for name in variants:
            tags=[name] if name=="b0" else [f"{name}_seed{s}" for s in seeds]
            ps=[root/"results/runs"/args.run/"controlled"/tag/f"sample{sample}_ratio{ratio}_users.npz"
                for tag in tags for sample in config["controlled"]["sampling_seeds"]]
            measures[name]=aggregate_users(ps,metric)
        draws=bootstrap_measures(measures,config["bootstrap_samples"],config["bootstrap_seed"])
        controlled["paired_differences"][str(ratio)]={}
        for name in ("b3_k2","b3_k4"):
            for base in ("b1","b2","b4"):
                controlled["paired_differences"][str(ratio)][f"{name}_minus_{base}"]={
                    "mean_difference":controlled["models"][name][str(ratio)]["mean"]-controlled["models"][base][str(ratio)]["mean"],
                    "paired_user_bootstrap_95_ci":ci(draws[name]-draws[base])}
    write_json(out/"controlled_summary.json",controlled)
    figures=root/"figures"; figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":140})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    labels=["B0","B1","B2","B3 K=2","B3 K=4","B4"]
    for ax,metric,title in zip(axes,("ndcg_at_10","minority_recall_at_10_user_macro"),("Overall ranking","Minority-interest recall")):
        values=[summary["models"][n]["metrics"][metric] for n in variants]
        means=np.array([v["mean"] for v in values])
        lower=means-np.array([v["user_bootstrap_95_ci"][0] for v in values])
        upper=np.array([v["user_bootstrap_95_ci"][1] for v in values])-means
        ax.bar(labels,means,color=["#8093a8","#8093a8","#4285a4","#dc8437","#bd5329","#6a9475"])
        ax.errorbar(labels,means,yerr=np.maximum(0,np.vstack([lower,upper])),fmt="none",color="#333333",capsize=3)
        ax.set_title(title); ax.set_ylabel("NDCG@10" if metric == "ndcg_at_10" else "Minority Recall@10 (user macro)"); ax.tick_params(axis="x",rotation=25)
        ax.set_ylim(0,max(means)*1.18)
    fig.suptitle("MIND-small test: seed means and 95% user-bootstrap intervals")
    fig.savefig(figures/"model_comparison.png"); fig.savefig(figures/"model_comparison.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5),layout="constrained")
    for name,label in zip(variants,labels):
        values=[controlled["models"][name][str(r)] for r in (50,80,90)]
        ax.errorbar([50,20,10],[v["mean"] for v in values],
                    yerr=[v["sample_std"] for v in values],marker="o",capsize=3,label=label)
    ax.set_xticks([50,20,10],["50:50","80:20","90:10"]); ax.invert_xaxis()
    ax.set_xlabel("Dominant : fixed target history ratio (20 items)")
    ax.set_ylabel("Fixed target-category Recall@10 (user macro)")
    ax.set_title("Same cohort and targets; history sampling averaged\nError bars: training-seed SD")
    ax.legend(ncol=3); fig.savefig(figures/"controlled_history.png"); fig.savefig(figures/"controlled_history.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
    for name,label in zip(variants,labels):
        ax.plot([10,20,30],[summary["models"][name]["threshold_sensitivity"][str(t)]["mean"] for t in (10,20,30)],marker="o",label=label)
    ax.set_xlabel("Minority category threshold (%)"); ax.set_ylabel("Minority Recall@10 (user macro)")
    ax.set_title("Threshold sensitivity; eligible users differ by threshold"); ax.legend(ncol=3)
    fig.savefig(figures/"threshold_sensitivity.png"); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    colors=["#355c7d","#6c9bb1","#9fbdc9","#d2dfe3"]
    for ax,name,title in zip(axes,("b3_k2","b3_k4"),("Two interest heads","Four interest heads")):
        diagnostics=summary["models"][name]["attention_diagnostics"]
        fractions=np.array([v["winning_head_fractions"] for v in diagnostics])
        bottom=np.zeros(len(seeds))
        for head in range(fractions.shape[1]):
            ax.bar([str(s) for s in seeds],fractions[:,head],bottom=bottom,
                   color=colors[head],label=f"Head {head+1}")
            bottom+=fractions[:,head]
        similarity=np.mean([v["mean_pairwise_attention_cosine"] for v in diagnostics])
        ax.set_title(f"{title}; attention cosine = {similarity:.5f}")
        ax.set_xlabel("Training seed"); ax.set_ylabel("Share of winning candidate matches")
        ax.set_ylim(0,1); ax.legend(ncol=2,fontsize=8)
    fig.suptitle("Test attention diagnostics (nonempty histories)")
    fig.savefig(figures/"attention_diagnostics.png"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout="constrained")
    names=["b2","b3_k2","b3_k4"]
    for ax,metric,label in zip(axes,("ndcg_at_10","minority_recall_at_10_user_macro"),("NDCG@10","Minority Recall@10")):
        values=[summary["models"][n]["metrics"][metric] for n in names]
        ax.errorbar([1,2,4],[v["mean"] for v in values],yerr=[v["sample_std"] for v in values],
                    marker="o",capsize=3,color="#355c7d")
        ax.set_xticks([1,2,4]); ax.set_xlim(0,5); ax.set_xlabel("Interest count K")
        ax.set_ylabel(label); ax.set_title("Seed means and training-seed SD")
        for k,name,value in zip([1,2,4],names,values):
            ax.annotate(f'{summary["models"][name]["trainable_parameters"]:,} params',
                        (k,value["mean"]),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8)
    fig.savefig(figures/"interest_count.png"); plt.close(fig)

    print(json.dumps({n:{m:v["mean"] for m,v in r["metrics"].items()} for n,r in summary["models"].items()},indent=2))


if __name__=="__main__":
    main()



