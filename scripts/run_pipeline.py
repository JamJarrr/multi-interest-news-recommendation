"""Run/resume the locked experiment stages with durable logs."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="main")
    parser.add_argument("--stage", choices=("all", "train", "evaluate", "controlled", "summarize"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    logdir = root / "results/runs" / args.run
    logdir.mkdir(parents=True, exist_ok=True)
    stages = {
        "train": ["src.training.run_interests", "train"],
        "evaluate": ["src.training.run_interests", "evaluate"],
        "controlled": ["src.evaluation.controlled"],
        "summarize": ["src.evaluation.summarize_interests"],
    }
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for stage, module_args in stages.items():
        if args.stage not in ("all", stage):
            continue
        command = [sys.executable, "-u", "-m", *module_args, "--root", ".", "--run", args.run]
        print(f"Starting {stage}: {' '.join(command)}", flush=True)
        with (logdir / f"{stage}.log").open("a", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
                if code:
                    raise SystemExit(code)

if __name__=="__main__":
    main()

