#!/usr/bin/env python3
"""Prepare verifier distillation parquet by augmenting RL parquet with ref_prompt.

Inputs:
- RL parquet(s): dapo_rl_train.parquet, dapo_rl_val.parquet
- Oracle solution JSONL files from train_problems/ and val_problems/

Outputs:
- dapo_distill_train.parquet
- optionally dapo_distill_val.parquet

If the RL parquet name carries a suffix such as `_k3`, the derived distill
output name preserves it by default.

By default, rows whose oracle solution is missing or empty are dropped with a
summary report, since they cannot support the teacher-conditioned ref prompt.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from verifier.prompt_templates import render_oracle_prompt


DISTILL_TEACHER_PROMPT_ID = "noleak_verdict_full_ostyle"  # Strategy Y6-style teacher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _normalize_problem_text(problem_text: str) -> str:
    return " ".join(problem_text.split())


def _parse_bins(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def derive_distill_output_path(rl_path: str, split: str) -> Path:
    rl_file = Path(rl_path)
    prefix = f"dapo_rl_{split}"
    suffix = ""
    if rl_file.name.startswith(prefix) and rl_file.suffix == ".parquet":
        suffix = rl_file.stem[len(prefix) :]
    return rl_file.with_name(f"dapo_distill_{split}{suffix}.parquet")


def infer_keep_per_class_from_rl_path(rl_path: str, split: str) -> int:
    rl_file = Path(rl_path)
    prefix = f"dapo_rl_{split}"
    if rl_file.name.startswith(prefix) and rl_file.suffix == ".parquet":
        suffix = rl_file.stem[len(prefix) :]
        match = re.search(r"(?:^|_)k(\d+)(?:_|$)", suffix)
        if match:
            return int(match.group(1))
    return 1


def load_oracle_solution_map(
    problems_dir: str, split: str, bins: Sequence[str]
) -> Dict[str, Dict[str, str]]:
    """Return map: bin_name -> normalized_problem -> oracle_solution."""
    by_bin: Dict[str, Dict[str, str]] = {}
    base = Path(problems_dir)

    for bin_name in bins:
        path = base / f"{split}_problems_{bin_name}.jsonl"
        if not path.exists():
            raise RuntimeError(f"Missing oracle problems file: {path}")

        by_problem: Dict[str, str] = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                problem = _normalize_problem_text(d.get("problem", ""))
                oracle_solution = d.get("gpt-5.2-low_oracle", "")
                if problem and oracle_solution and problem not in by_problem:
                    by_problem[problem] = oracle_solution

        by_bin[bin_name] = by_problem

    return by_bin


def load_rl_rows(path: str) -> List[dict]:
    table = pq.read_table(path)
    rows: List[dict] = []

    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}

        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        reward_model = row.get("reward_model", {})
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)

        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        rows.append(
            {
                "prompt": _to_native(prompt),
                "data_source": str(row.get("data_source", "verifier")),
                "reward_model": _to_native(reward_model),
                "extra_info": _to_native(extra_info),
            }
        )

    return rows


def build_distill_rows(
    rl_rows: Sequence[dict],
    oracle_by_bin: Dict[str, Dict[str, str]],
    missing_oracle_policy: str,
) -> tuple[List[dict], int]:
    rows: List[dict] = []
    missing_oracle = 0

    for r in rl_rows:
        extra = r["extra_info"]
        bin_name = extra.get("bin")
        problem = extra.get("problem", "")
        solution = extra.get("solution", "")
        verdict = r["reward_model"].get("ground_truth", "")

        if not bin_name or not problem or not solution or not verdict:
            raise RuntimeError(
                "RL row missing required fields in extra_info/reward_model; "
                "please regenerate RL data with current prepare_verifier_data.py"
            )

        oracle_solution = oracle_by_bin.get(bin_name, {}).get(_normalize_problem_text(problem), "")
        if not oracle_solution:
            missing_oracle += 1
            if missing_oracle_policy == "error":
                continue
            continue

        ref_prompt_text = render_oracle_prompt(
            DISTILL_TEACHER_PROMPT_ID,
            problem=problem,
            solution=solution,
            verdict=verdict,
            oracle_solution=oracle_solution,
        )

        out = {
            "prompt": r["prompt"],
            "data_source": r["data_source"],
            "reward_model": r["reward_model"],
            "extra_info": r["extra_info"],
            "ref_prompt": [{"role": "user", "content": ref_prompt_text}],
        }
        rows.append(out)

    if missing_oracle > 0 and missing_oracle_policy == "error":
        raise RuntimeError(
            f"Missing oracle solutions for {missing_oracle} rows while building distill data."
        )

    return rows, missing_oracle


def write_distill_parquet(rows: Sequence[dict], output_path: Path) -> int:
    table = pa.table(
        {
            # Keep nested columns native so verl DataLoader receives dict/list types.
            "prompt": pa.array([_to_native(r["prompt"]) for r in rows]),
            "data_source": pa.array([r["data_source"] for r in rows]),
            "reward_model": pa.array([_to_native(r["reward_model"]) for r in rows]),
            "extra_info": pa.array([_to_native(r["extra_info"]) for r in rows]),
            "ref_prompt": pa.array([_to_native(r["ref_prompt"]) for r in rows]),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Prepare verifier distill parquet from RL parquet")
    p.add_argument("--rl_train_file", type=str, default="verifier/data/dapo_rl_train.parquet")
    p.add_argument("--rl_val_file", type=str, default="verifier/data/dapo_rl_val.parquet")
    p.add_argument(
        "--train_problems_dir",
        type=str,
        default="retrieval-analogy/data/train_problems",
    )
    p.add_argument(
        "--val_problems_dir",
        type=str,
        default="retrieval-analogy/data/val_problems",
    )
    p.add_argument("--bins", type=str, default="0.0_0.0,0.0_0.2")
    p.add_argument(
        "--output_train_file",
        type=str,
        default="auto",
    )
    p.add_argument(
        "--output_val_file",
        type=str,
        default=None,
        help=(
            "Optional output path for distill val parquet. Leave unset to keep "
            "using RL val data, or pass 'auto' to mirror the rl_val_file name."
        ),
    )
    p.add_argument(
        "--missing_oracle_policy",
        type=str,
        default="drop",
        choices=["drop", "error"],
        help="How to handle rows whose oracle solution is missing or empty.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    bins = _parse_bins(args.bins)
    train_keep_per_class = infer_keep_per_class_from_rl_path(
        args.rl_train_file, "train"
    )
    val_skip_message = None
    train_output_path = (
        derive_distill_output_path(args.rl_train_file, "train")
        if args.output_train_file == "auto"
        else Path(args.output_train_file)
    )
    val_output_path = None
    if train_keep_per_class > 1:
        if args.output_val_file is not None:
            val_skip_message = (
                "Distill val:   skipped because "
                f"{Path(args.rl_train_file).name} uses train-only K={train_keep_per_class}; "
                "keep using verifier/data/dapo_rl_val.parquet for validation."
            )
    else:
        if args.output_val_file == "auto":
            val_output_path = derive_distill_output_path(args.rl_val_file, "val")
        elif args.output_val_file:
            val_output_path = Path(args.output_val_file)

    train_oracle = load_oracle_solution_map(args.train_problems_dir, "train", bins)

    train_rl_rows = load_rl_rows(args.rl_train_file)

    train_distill_rows, train_missing = build_distill_rows(
        train_rl_rows,
        train_oracle,
        missing_oracle_policy=args.missing_oracle_policy,
    )

    n_train = write_distill_parquet(train_distill_rows, train_output_path)

    print(f"Distill train: {n_train} -> {train_output_path}")
    if val_output_path is not None:
        val_oracle = load_oracle_solution_map(args.val_problems_dir, "val", bins)
        val_rl_rows = load_rl_rows(args.rl_val_file)
        val_distill_rows, val_missing = build_distill_rows(
            val_rl_rows,
            val_oracle,
            missing_oracle_policy=args.missing_oracle_policy,
        )
        n_val = write_distill_parquet(val_distill_rows, val_output_path)
        print(f"Distill val:   {n_val} -> {val_output_path}")
    else:
        val_missing = None
        if val_skip_message is not None:
            print(val_skip_message)
        else:
            print("Distill val:   skipped (use verifier/data/dapo_rl_val.parquet for validation)")

    if args.missing_oracle_policy == "drop":
        print(
            "Dropped rows with missing/empty oracle solution: "
            f"train={train_missing}"
        )
        if val_missing is not None:
            print(f"Dropped val rows with missing/empty oracle solution: val={val_missing}")


if __name__ == "__main__":
    main()
