#!/usr/bin/env python3
"""Prepare RL verifier training data from DAPO train/val rollouts.

Workflow:
1) Generate raw rollouts for each split (train/val)
2) Balance per problem:
   - keep up to K random CORRECT rollouts
   - keep up to K random INCORRECT rollouts
3) Write RL parquet(s):
   - dapo_rl_train.parquet
   - dapo_rl_val.parquet

Distillation data is generated separately by:
- verifier/prepare_verifier_distill_data.py
"""

import argparse
import gc
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from examples.sdpo_trainer.math_reward import compute_score
from verifier.prompt_templates import (
    VERIFY_UNCONDITIONAL_PROMPT_ID,
    render_oracle_prompt,
)


REPROMPT_WITH_FEEDBACK = (
    "A reviewer found issues with your solution:\n\n"
    "{feedback}\n\n"
    "Please solve the problem again, fixing the identified errors. "
    "Put your final answer inside \\boxed{{}}."
)
GENERIC_REPROMPT_FEEDBACK = "Your solution appears to be incorrect."


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


def load_parquet_samples(path: str) -> List[dict]:
    table = pq.read_table(path)
    samples = []
    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        reward_model = row.get("reward_model", {})
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)
        samples.append(
            {
                "prompt": _to_native(prompt),
                "data_source": str(row.get("data_source", "")),
                "reward_model": _to_native(reward_model),
            }
        )
    return samples


def get_problem_text(sample: dict) -> str:
    for msg in reversed(sample["prompt"]):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def _problem_hash(problem_text: str) -> str:
    text = " ".join(problem_text.split())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _parse_bins(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _normalize_problem_text(problem_text: str) -> str:
    return " ".join(problem_text.split())


def parse_oracle_response(text: str) -> tuple[str, Optional[str]]:
    """Parse VERDICT and FEEDBACK/FINDINGS from binary verifier output."""
    verdict = None
    feedback = None

    for line in text.split("\n"):
        line_stripped = line.strip()
        if line_stripped.upper().startswith("VERDICT:"):
            verdict_text = line_stripped[len("VERDICT:"):].strip().upper()
            if "INCORRECT" in verdict_text:
                verdict = "INCORRECT"
            elif "CORRECT" in verdict_text:
                verdict = "CORRECT"
            break

    feedback_match = re.search(r"(?:FEEDBACK|FINDINGS):\s*(.*)", text, re.DOTALL)
    if feedback_match:
        feedback = feedback_match.group(1).strip()
        if feedback.lower() in ("none", "none.", "n/a", "n/a."):
            feedback = None

    if verdict is None:
        verdict = "INCORRECT" if "incorrect" in text.lower() else "CORRECT"

    return verdict, feedback


def resolve_dapo_parquets(data_dir: str, bins: Sequence[str], split: str) -> List[Path]:
    """Resolve DAPO parquet paths for each bin and split."""
    base = Path(data_dir)
    files: List[Path] = []
    for bin_name in bins:
        candidates = [
            base / f"{split}_{bin_name}.parquet",
            base / f"{split}_parquets" / f"{split}_{bin_name}.parquet",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            print(
                f"Warning: missing DAPO {split} parquet for bin={bin_name} under {base}"
            )
            continue
        files.append(found)
    return files


def _bin_from_parquet_path(parquet_path: Path) -> str:
    stem = parquet_path.stem
    if stem.startswith("train_"):
        return stem[len("train_") :]
    if stem.startswith("val_"):
        return stem[len("val_") :]
    return stem


# ---------------------------------------------------------------------------
# Collect raw rollouts
# ---------------------------------------------------------------------------


def collect_from_rollouts(
    model_name: str,
    llm,
    tokenizer,
    gen_params,
    parquet_paths: Sequence[Path],
    rollout_n: int,
    seed: int,
    max_problems_per_bin: Optional[int],
) -> List[dict]:
    """Generate fresh rollouts with vLLM and collect raw pairs."""
    rng = random.Random(seed)
    pairs = []

    for parquet_path in parquet_paths:
        bin_name = _bin_from_parquet_path(parquet_path)
        samples = load_parquet_samples(str(parquet_path))
        if max_problems_per_bin is not None and len(samples) > max_problems_per_bin:
            samples = rng.sample(samples, max_problems_per_bin)

        prompts = [apply_chat_template(tokenizer, s["prompt"]) for s in samples]
        print(
            f"  Generating {len(prompts)} x {rollout_n} rollouts for {bin_name} "
            f"from {parquet_path}..."
        )
        outputs = llm.generate(prompts, gen_params)

        for prob_i, (sample, out) in enumerate(zip(samples, outputs)):
            problem_text = get_problem_text(sample)
            gt = sample["reward_model"].get("ground_truth", "")

            for chain_idx, completion in enumerate(out.outputs):
                solution = completion.text
                true_reward = compute_score(
                    data_source=sample["data_source"],
                    solution_str=solution,
                    ground_truth=gt,
                )
                pairs.append(
                    {
                        "problem": problem_text,
                        "solution": solution,
                        "ground_truth_answer": gt,
                        "true_reward": float(true_reward),
                        "bin": bin_name,
                        "problem_idx": int(prob_i),
                        "chain_idx": int(chain_idx),
                        "strategy": "rollout",
                        "source_run": model_name,
                        "problem_hash": _problem_hash(problem_text),
                    }
                )
    return pairs


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------


def _is_correct(pair: dict) -> bool:
    return float(pair["true_reward"]) > 0.5


def summarize_problem_mix(pairs: Sequence[dict]) -> Dict[str, int]:
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for p in pairs:
        key = (p["bin"], p["problem_hash"])
        grouped[key].append(p)

    both = only_correct = only_incorrect = 0
    for items in grouped.values():
        n_correct = sum(1 for x in items if _is_correct(x))
        n_incorrect = len(items) - n_correct
        if n_correct > 0 and n_incorrect > 0:
            both += 1
        elif n_correct > 0:
            only_correct += 1
        else:
            only_incorrect += 1

    return {
        "problems": len(grouped),
        "both": both,
        "only_correct": only_correct,
        "only_incorrect": only_incorrect,
    }


def _sample_up_to(
    items: Sequence[dict], limit: int, rng: random.Random
) -> List[dict]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    return rng.sample(list(items), limit)


def balance_pairs_per_problem(
    pairs: Sequence[dict],
    seed: int,
    max_keep_per_class_per_problem: int,
) -> List[dict]:
    """Keep up to K correct and up to K incorrect rollouts per problem."""
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for p in pairs:
        grouped[(p["bin"], p["problem_hash"])].append(p)

    rng = random.Random(seed)
    out: List[dict] = []

    # Sorted keys for deterministic iteration across runs.
    for key in sorted(grouped.keys()):
        items = grouped[key]
        correct = [x for x in items if _is_correct(x)]
        incorrect = [x for x in items if not _is_correct(x)]

        out.extend(_sample_up_to(correct, max_keep_per_class_per_problem, rng))
        out.extend(_sample_up_to(incorrect, max_keep_per_class_per_problem, rng))

    return out


# ---------------------------------------------------------------------------
# vLLM lifecycle
# ---------------------------------------------------------------------------


def init_vllm_engine(
    model_name: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    seed: int,
    rollout_n: int,
):
    from vllm import LLM, SamplingParams

    print(f"\nInitializing vLLM: {model_name}")
    print(f"Parallel config: TP={tensor_parallel_size}")

    llm_kwargs = dict(
        model=model_name,
        trust_remote_code=True,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    gen_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096,
        n=rollout_n,
    )
    followup_gen_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096,
        n=1,
    )
    verifier_gen_params = SamplingParams(
        temperature=0.3,
        top_p=1.0,
        max_tokens=512,
        n=1,
    )
    return llm, tokenizer, gen_params, followup_gen_params, verifier_gen_params


def cleanup_vllm_engine(llm) -> None:
    # Best-effort cleanup so repeated runs start from a clean GPU state.
    try:
        if hasattr(llm, "close") and callable(llm.close):
            llm.close()
    except Exception:
        pass
    try:
        eng = getattr(llm, "llm_engine", None)
        if eng is not None and hasattr(eng, "shutdown") and callable(eng.shutdown):
            eng.shutdown()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------


def build_rl_parquet(
    pairs: Sequence[dict],
    output_path: Path,
    split: str,
) -> int:
    rows = []
    for pair in pairs:
        is_correct = _is_correct(pair)
        verdict_gt = "CORRECT" if is_correct else "INCORRECT"

        prompt_text = render_oracle_prompt(
            VERIFY_UNCONDITIONAL_PROMPT_ID,
            problem=pair["problem"],
            solution=pair["solution"],
        )
        extra_info = {
            "bin": pair["bin"],
            "problem_idx": pair["problem_idx"],
            "problem_hash": pair["problem_hash"],
            "chain_idx": pair["chain_idx"],
            "strategy": pair["strategy"],
            "source_run": pair["source_run"],
            "true_reward": pair["true_reward"],
            "ground_truth_answer": pair["ground_truth_answer"],
            # Stored for separate distill conversion.
            "problem": pair["problem"],
            "solution": pair["solution"],
        }

        data_source = "verifier"

        row = {
            "prompt": [{"role": "user", "content": prompt_text}],
            "data_source": data_source,
            "reward_model": {"style": "rule", "ground_truth": verdict_gt},
            "extra_info": extra_info,
        }
        rows.append(row)

    table = pa.table(
        {
            # Keep nested columns native so verl DataLoader receives dict/list types.
            "prompt": pa.array([_to_native(r["prompt"]) for r in rows]),
            "data_source": pa.array([r["data_source"] for r in rows]),
            "reward_model": pa.array([_to_native(r["reward_model"]) for r in rows]),
            "extra_info": pa.array([_to_native(r["extra_info"]) for r in rows]),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return len(rows)


def print_stats(name: str, pairs: Sequence[dict]):
    n_pairs = len(pairs)
    n_correct = sum(1 for p in pairs if _is_correct(p))
    n_problems = len({(p["bin"], p["problem_hash"]) for p in pairs})
    print(
        f"{name}: pairs={n_pairs}, problems={n_problems}, "
        f"correct={n_correct}, incorrect={n_pairs - n_correct}"
    )
    by_bin = defaultdict(int)
    for p in pairs:
        by_bin[p["bin"]] += 1
    print("  per-bin:", dict(sorted(by_bin.items())))


def rl_output_paths(
    output_dir: Path,
    max_keep_per_class_per_problem: int,
) -> Tuple[Path, Path]:
    suffix_parts: List[str] = []
    if max_keep_per_class_per_problem > 1:
        suffix_parts.append(f"k{max_keep_per_class_per_problem}")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    return (
        output_dir / f"dapo_rl_train{suffix}.parquet",
        output_dir / f"dapo_rl_val{suffix}.parquet",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Prepare RL verifier training data for verl")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument(
        "--data_dir",
        type=str,
        default="retrieval-analogy/data",
        help="Base directory containing DAPO train/val parquets.",
    )
    p.add_argument(
        "--bins",
        type=str,
        default="0.0_0.0,0.0_0.2",
        help="Comma-separated DAPO bins.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "val", "both"],
        help="Which split(s) to generate.",
    )
    p.add_argument(
        "--max_problems_per_bin",
        type=int,
        default=None,
        help="Optional cap for sampled problems per split/bin.",
    )
    p.add_argument("--rollout_n", type=int, default=16)
    p.add_argument(
        "--max_keep_per_class_per_problem",
        type=int,
        default=1,
        help=(
            "Keep up to this many correct and this many incorrect rollouts per "
            "problem after generation. When >1, output filenames gain a _kN suffix."
        ),
    )
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="verifier/data")
    return p.parse_args()


def main():
    args = parse_args()
    if args.max_keep_per_class_per_problem < 1:
        raise ValueError("--max_keep_per_class_per_problem must be >= 1")
    random.seed(args.seed)
    bins = _parse_bins(args.bins)

    requested_splits: List[str] = []
    if args.split in {"train", "both"}:
        requested_splits.append("train")
    if args.split in {"val", "both"}:
        if args.max_keep_per_class_per_problem > 1:
            print(
                "Skipping VAL because "
                f"--max_keep_per_class_per_problem={args.max_keep_per_class_per_problem} "
                "is train-only; keep using verifier/data/dapo_rl_val.parquet "
                "for validation."
            )
        else:
            requested_splits.append("val")

    if not requested_splits:
        print("Nothing to do.")
        return

    parquet_paths_by_split: Dict[str, List[Path]] = {}
    for split in requested_splits:
        paths = resolve_dapo_parquets(args.data_dir, bins, split=split)
        if not paths:
            raise RuntimeError(f"No DAPO {split} parquets resolved.")
        parquet_paths_by_split[split] = paths

    raw_pairs_by_split: Dict[str, List[dict]] = {}

    llm, tokenizer, gen_params, _, _ = init_vllm_engine(
        model_name=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        rollout_n=args.rollout_n,
    )
    try:
        for split in requested_splits:
            print(f"Collecting {split.upper()} raw rollouts...")
            raw_pairs_by_split[split] = collect_from_rollouts(
                model_name=args.model,
                llm=llm,
                tokenizer=tokenizer,
                gen_params=gen_params,
                parquet_paths=parquet_paths_by_split[split],
                rollout_n=args.rollout_n,
                seed=args.seed,
                max_problems_per_bin=args.max_problems_per_bin,
            )
    finally:
        cleanup_vllm_engine(llm)

    # Balance and write RL parquet(s)
    balanced_pairs_by_split: Dict[str, List[dict]] = {}
    for split in requested_splits:
        raw_pairs = raw_pairs_by_split[split]
        if not raw_pairs:
            raise RuntimeError(f"No raw rollouts available for split={split}.")

        mix = summarize_problem_mix(raw_pairs)
        print(
            f"{split.upper()} raw mix: problems={mix['problems']} both={mix['both']} "
            f"only_correct={mix['only_correct']} only_incorrect={mix['only_incorrect']}"
        )

        # Offset seed by split for deterministic but different sampling.
        split_seed = args.seed + (0 if split == "train" else 1000003)
        balanced = balance_pairs_per_problem(
            raw_pairs,
            seed=split_seed,
            max_keep_per_class_per_problem=args.max_keep_per_class_per_problem,
        )
        balanced_pairs_by_split[split] = balanced
        print_stats(f"{split.upper()} balanced", balanced)

    output_dir = Path(args.output_dir)
    rl_train_path, rl_val_path = rl_output_paths(
        output_dir,
        max_keep_per_class_per_problem=args.max_keep_per_class_per_problem,
    )

    if "train" in balanced_pairs_by_split:
        n = build_rl_parquet(
            balanced_pairs_by_split["train"],
            rl_train_path,
            split="train",
        )
        print(f"\nRL train: {n} -> {rl_train_path}")
    if "val" in balanced_pairs_by_split:
        n = build_rl_parquet(
            balanced_pairs_by_split["val"],
            rl_val_path,
            split="val",
        )
        print(f"RL val:   {n} -> {rl_val_path}")

    print(
        "\nNext step for distill data: python verifier/prepare_verifier_distill_data.py"
    )


if __name__ == "__main__":
    main()
