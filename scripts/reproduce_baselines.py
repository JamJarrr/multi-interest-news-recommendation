"""Reproduce the parameter-free baseline and one original CPU training seed."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    result = ROOT / "results/reproduction"
    result.mkdir(parents=True, exist_ok=True)
    commands = [
        ["src.training.run_b0", "--results-dir", "results/reproduction"],
        ["src.training.train_b1", "--seed", "42", "--device", "cpu",
         "--results-dir", "results/reproduction", "--checkpoint-dir", "checkpoints/reproduction"],
    ]
    for args, filename in zip(commands, ("b0_metrics.json", "b1_seed42_metrics.json"), strict=True):
        if not (result / filename).exists():
            subprocess.run([sys.executable, "-u", "-m", *args], cwd=ROOT, env=env, check=True)
    local_manifest=json.loads((ROOT/"results/local_data/source_manifest.json").read_text())
    original_manifest=json.loads((ROOT/"results/reference/data_manifest.json").read_text())
    archive_matches=all(local_manifest[k]["sha256"]==v["sha256"] for k,v in original_manifest.items())
    report = {"description":"Original B0 and original B1 seed42 on the locally regenerated CUDA-encoded cache; historical summaries are preserved.",
              "audit_matches":json.loads((ROOT/"results/data_audit.json").read_text()) == json.loads((ROOT/"results/local_data/data_audit.json").read_text()),
              "archive_sha256_matches":archive_matches, "numerical_tolerance":1e-4,"runs":{}}
    for filename in ("b0_metrics.json","b1_seed42_metrics.json"):
        actual=json.loads((result/filename).read_text())
        previous=json.loads((ROOT/"results/reference"/filename).read_text())
        differences={}
        for split in ("validation","test"):
            differences[split]={key:actual["metrics"][split][key]-previous["metrics"][split][key]
                                for key in ("ndcg_at_5","ndcg_at_10","recall_at_10","minority_recall_at_10_user_macro")}
        maximum=max(abs(x) for values in differences.values() for x in values.values())
        report["runs"][filename]={"delta_local_minus_reference":differences,
                                 "max_absolute_difference":maximum,"within_tolerance":maximum<=1e-4}
    (result/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()


