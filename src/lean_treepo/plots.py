from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Lean TreePO metrics from JSONL metric exports.")
    parser.add_argument("--runs-dir", default="/workspace/outputs/treepo-runs")
    parser.add_argument("--output-dir", default=None)
    return parser


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    import matplotlib.pyplot as plt
    import pandas as pd

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir or runs_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(runs_dir.glob("**/metrics.jsonl"))
    if not metric_files:
        print(f"No metrics.jsonl files found under {runs_dir}.")
        return

    rows = []
    for path in metric_files:
        method = path.parent.name
        for row in load_jsonl(path):
            row["method"] = method
            rows.append(row)
    df = pd.DataFrame(rows)

    for metric in ["lean/accuracy", "entropy", "completions/mean_length", "tree/tokenps", "tree/trajps", "reward"]:
        if metric not in df:
            continue
        plt.figure()
        for method, group in df.groupby("method"):
            x = group["step"] if "step" in group else range(len(group))
            plt.plot(x, group[metric], label=method)
        plt.title(metric)
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric.replace('/', '_')}.png")
        plt.close()
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
