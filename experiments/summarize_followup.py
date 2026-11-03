"""Summarize the locked follow-up without changing training/selection."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.followup import ROOT, OUT, LOCAL, SEEDS, identifier, tag, read
from src.training.common import write_json
from src.evaluation.summarize_interests import stats, aggregate_users, bootstrap_measures, ci

PAIRS=[("b2_tuned","b2_original"),("b3_scaled","b3_original"),
       ("b3_scaled","b2_matched"),("b3_scaled","b2_tuned"),
       ("b3_diversity","b3_scaled"),("b3_diversity_diagnostic","b3_scaled"),
       ("b3_diversity","b2_tuned"),("category_groups","b0"),
       ("b3_original","b2_original")]
LABELS={"b2_original":"B2 original","b3_original":"B3 original",
        "b2_tuned":"B2 tuned","b2_matched":"B2 matched scale",
        "b3_scaled":"B3 scaled","b3_diversity":"B3 selected diversity",
        "b3_diversity_diagnostic":"B3 diversity 0.1",
        "category_groups":"Category groups (extra metadata)","b0":"B0 frozen mean"}


def main():
    selection=read(OUT/"selection.json")
    roles={name:[tag(s,seed) for seed in SEEDS] for name,s in selection["roles"].items()}
    roles.update(b0=["b0"],category_groups=["category_groups"])
    result={"selection":selection,"models":{},"paired_differences":{},
            "limitations":["Exploratory after original test inspection.",
             "95% paired user-bootstrap intervals conditional on fitted models; seed SD separate.",
             "No multiple-comparison correction; screening used only one seed.",
             "Category groups use additional metadata and variable interest count, not a capacity-matched comparison."]}
    metrics=["ndcg_at_10","minority_recall_at_10_user_macro"]
    for name,tags in roles.items():
        records=[read(OUT/(t+"_test.json")) for t in tags]
        result["models"][name]={"full":{},"last50":{},
            "diagnostics":[r["diagnostics"] for r in records],
            "coverage":{scope:records[0][key]["minority_target_users"]
                        for scope,key in [("full","metrics"),("last50","last50_metrics")]}}
        for scope,key in [("full","metrics"),("last50","last50_metrics")]:
            for metric in metrics:
                result["models"][name][scope][metric]=stats([r[key][metric] for r in records])
    for scope,suffix in [("full","test_users.npz"),("last50","test_last50_users.npz")]:
        result["paired_differences"][scope]={}
        for metric in metrics:
            measures={name:aggregate_users([LOCAL/t/suffix for t in tags],metric)
                      for name,tags in roles.items()}
            draws=bootstrap_measures(measures,1000,8675309)
            for name in roles:
                result["models"][name][scope][metric]["ci95"]=ci(draws[name])
            for a,b in PAIRS:
                key=a+"_minus_"+b
                result["paired_differences"][scope].setdefault(key,{})[metric]={
                    "difference":result["models"][a][scope][metric]["mean"]-result["models"][b][scope][metric]["mean"],
                    "ci95":ci(draws[a]-draws[b])}
    controlled={"design":read(OUT/"controlled_design.json"),"models":{},"paired_differences":{}}
    metric="target_category_recall_at_10_user_macro"
    for name,tags in roles.items():
        controlled["models"][name]={}
        for ratio in [50,80,90]:
            values=[[read(OUT/f"{t}_sample{s}_ratio{ratio}.json")["metrics"][metric]
                     for s in [11,29,47]] for t in tags]
            controlled["models"][name][str(ratio)]={
                **stats([np.mean(v) for v in values]),"by_training_seed_and_sampling_seed":values}
    for ratio in [50,80,90]:
        measures={name:aggregate_users([LOCAL/t/f"sample{s}_ratio{ratio}_users.npz"
                                       for t in tags for s in [11,29,47]],metric)
                  for name,tags in roles.items()}
        draws=bootstrap_measures(measures,1000,8675309)
        controlled["paired_differences"][str(ratio)]={}
        for name in roles:
            controlled["models"][name][str(ratio)]["ci95"]=ci(draws[name])
            controlled["models"][name][str(ratio)]["target_users"]=int(measures[name][2].sum())
        for a,b in PAIRS:
            controlled["paired_differences"][str(ratio)][a+"_minus_"+b]={
                "difference":controlled["models"][a][str(ratio)]["mean"]-controlled["models"][b][str(ratio)]["mean"],
                "ci95":ci(draws[a]-draws[b])}
    result["controlled"]=controlled
    write_json(OUT/"summary.json",result)
    with (OUT/"comparison.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["role","spec","history_scope","metric","mean","seed_sd","ci_low","ci_high"])
        for name,record in result["models"].items():
            for scope in ["full","last50"]:
                for metric,v in record[scope].items():
                    w.writerow([name,identifier(selection["roles"][name]) if name in selection["roles"] else name,
                                scope,metric,v["mean"],v["sample_std"],*v["ci95"]])
    figdir=ROOT/"figures/followup"
    figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    screen=selection["screening"]
    for heads,color in [(1,"#376b85"),(4,"#b05a32")]:
        runs=sorted([r for r in screen if r["spec"]["heads"]==heads and r["spec"]["diversity_weight"]==0],
                    key=lambda r:r["spec"]["attention_scale"])
        for ax,metric,title in zip(axes[:2],metrics,["Validation NDCG@10","Validation minority Recall@10"]):
            ax.plot([r["spec"]["attention_scale"] for r in runs],[r["metrics"][metric] for r in runs],
                    marker="o",label=f"K={heads}",color=color)
            ax.set_xscale("log",base=2);ax.set_xticks([1,8,32],[1,8,32]);ax.set_xlabel("Attention logit multiplier")
            ax.set_ylabel(title);ax.legend()
        axes[2].plot([r["spec"]["attention_scale"] for r in runs],
                     [r["diagnostics"]["normalized_attention_entropy"] for r in runs],
                     marker="o",label=f"K={heads}",color=color)
    axes[2].set_xscale("log",base=2);axes[2].set_xticks([1,8,32],[1,8,32])
    axes[2].set_xlabel("Attention logit multiplier");axes[2].set_ylabel("Normalized attention entropy (1 = uniform)")
    axes[2].legend()
    fig.suptitle("Validation screening (seed 42); frozen data and training budget")
    fig.savefig(figdir/"validation_screening.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout="constrained")
    pairs=[("b3_original","b2_original"),("b3_diversity","b3_scaled"),
           ("b3_diversity","b2_tuned"),("category_groups","b0")]
    labels=[LABELS[a]+" minus "+LABELS[b] for a,b in pairs]
    for ax,metric,title in zip(axes,metrics,["NDCG@10 difference","Minority Recall@10 difference"]):
        for i,(a,b) in enumerate(pairs):
            v=result["paired_differences"]["full"][a+"_minus_"+b][metric]
            ax.plot(v["ci95"],[i,i],color="#376b85")
            ax.plot(v["difference"],i,"o",color="#376b85")
        ax.axvline(0,color="grey",linestyle="--")
        ax.set_yticks(range(len(labels)),labels if ax is axes[0] else [])
        ax.invert_yaxis();ax.set_xlabel(title)
    fig.suptitle("Exploratory test comparisons: 95% paired user-bootstrap intervals\nConditional on fitted seeds; no multiple-comparison correction")
    fig.savefig(figdir/"paired_differences.png");fig.savefig(figdir/"paired_differences.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    for name in ["b0","b2_matched","b3_scaled","b3_diversity","category_groups"]:
        vals=[controlled["models"][name][str(r)] for r in [50,80,90]]
        ax.errorbar([0,1,2],[v["mean"] for v in vals],yerr=[v["sample_std"] for v in vals],
                    marker="o",capsize=3,label=LABELS[name])
    ax.set_xticks([0,1,2],["50:50","80:20","90:10"])
    ax.set_xlabel("Dominant : fixed target history ratio (20 events)")
    ax.set_ylabel("Fixed target-category Recall@10 (user macro)")
    ax.set_title("Same cohort and targets; 3 history samples averaged\nError bars: training-seed SD")
    ax.legend(fontsize=9,ncol=2)
    fig.savefig(figdir/"controlled_history.png");fig.savefig(figdir/"controlled_history.pdf");plt.close(fig)
    print(json.dumps({n:{"ndcg":r["full"]["ndcg_at_10"]["mean"],
                         "minority":r["full"]["minority_recall_at_10_user_macro"]["mean"]}
                      for n,r in result["models"].items()},indent=2))


if __name__=="__main__":
    main()