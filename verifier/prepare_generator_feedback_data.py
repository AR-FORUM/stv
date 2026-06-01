#!/usr/bin/env python3
"""Prepare offline generator-feedback training data.

Workflow:
1) Generate round-0 rollouts from the RL generator
2) Run trained verifier (strategy O = identify_mistake) on all rollouts
3) Filter to verifier-INCORRECT rollouts. Optionally also add false accepts
   (ground-truth wrong, verifier says CORRECT) with the generic reprompt.
4) Build multi-turn training parquet:
   [user: problem, assistant: solution, user: feedback reprompt]

The resulting parquet is compatible with standard GRPO + math_reward.py.

Output:
- verifier/data/dapo_generator_feedback_train.parquet
- verifier/data/dapo_generator_feedback_val.parquet
"""

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from verifier.prepare_verifier_data import (
    GENERIC_REPROMPT_FEEDBACK,
    REPROMPT_WITH_FEEDBACK,
    _to_native,
    apply_chat_template,
    cleanup_vllm_engine,
    collect_from_rollouts,
    init_vllm_engine,
    parse_oracle_response,
    resolve_dapo_parquets,
)
from verifier.prompt_templates import (
    VERIFY_UNCONDITIONAL_PROMPT_ID,
    get_oracle_template,
)


def check_answer_leakage(feedback, ground_truth):
    """Heuristic check for whether verifier feedback leaked the answer.

    Regex/substring based: catches the ground-truth answer stated literally or
    via a few "the answer is X" phrasings. It cannot detect paraphrased or
    implicit leakage, and is only used to *prefer* non-leaky feedback when
    sampling (see --prefer_non_leaky_feedback)."""
    if not feedback or not ground_truth:
        return False
    gt = str(ground_truth).strip()
    gt = re.sub(r"^\$+", "", gt)
    gt = re.sub(r"\$+$", "", gt)
    gt = gt.strip()
    if not gt:
        return False

    gt_esc = re.escape(gt)
    patterns = [
        rf"(?:correct|right|actual|true)\s+(?:answer|result|value)\s+"
        rf"(?:is|should be|equals?)\s*(?:\$\s*)?{gt_esc}",
        rf"(?:the\s+)?answer\s+(?:is|should be|equals?)\s*(?:\$\s*)?"
        rf"{gt_esc}(?:\s|\$|[,.\);]|$)",
        rf"is\s+(?:actually|in fact)\s+(?:\$\s*)?{gt_esc}(?:\s|\$|[,.\);]|$)",
        rf"which\s+(?:is|gives|equals?)\s+(?:\$\s*)?{gt_esc}(?:\s|\$|[,.\);]|$)",
    ]
    for pattern in patterns:
        if re.search(pattern, feedback, re.IGNORECASE):
            return True

    if len(gt) >= 3 and gt in feedback:
        return True
    gt_nospace = gt.replace(" ", "")
    feedback_nospace = feedback.replace(" ", "")
    if len(gt_nospace) >= 4 and gt_nospace in feedback_nospace:
        return True

    return False


# ---------------------------------------------------------------------------
# Verification pass
# ---------------------------------------------------------------------------


def run_verifier_on_pairs(
    verifier_model: str,
    pairs: List[dict],
    verifier_prompt_id: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    seed: int,
) -> List[dict]:
    """Run verifier on all pairs and return augmented pairs."""
    verifier_template = get_oracle_template(verifier_prompt_id)

    llm, tokenizer, _, _, verifier_gen_params = init_vllm_engine(
        model_name=verifier_model,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        rollout_n=1,
    )

    # Build verifier prompts
    verifier_prompts = []
    for pair in pairs:
        prompt_kwargs = build_verifier_prompt_kwargs(pair, verifier_template)
        prompt_text = verifier_template.format(**prompt_kwargs)
        verifier_prompts.append(
            apply_chat_template(tokenizer, [{"role": "user", "content": prompt_text}])
        )

    print(
        f"  Running verifier on {len(verifier_prompts)} rollouts "
        f"with prompt_id={verifier_prompt_id}..."
    )
    verifier_outputs = llm.generate(verifier_prompts, verifier_gen_params)
    cleanup_vllm_engine(llm)

    augmented = []
    for pair, vout in zip(pairs, verifier_outputs):
        verdict, feedback = parse_oracle_response(vout.outputs[0].text)
        aug = dict(pair)
        aug["verifier_verdict"] = verdict
        aug["verifier_feedback"] = feedback
        aug["verifier_prompt_id"] = verifier_prompt_id
        aug["verifier_feedback_leaked_answer"] = check_answer_leakage(
            feedback or "", pair.get("ground_truth_answer", "")
        )
        augmented.append(aug)

    return augmented


def build_verifier_prompt_kwargs(
    pair: dict,
    verifier_template: str,
) -> dict:
    """Build prompt kwargs for the unprivileged verifier prompt."""
    kwargs = {
        "problem": pair["problem"],
        "solution": pair["solution"],
    }
    if "{ground_truth}" in verifier_template:
        kwargs["ground_truth"] = pair.get("ground_truth_answer", "")
    if "{verdict}" in verifier_template:
        kwargs["verdict"] = (
            "CORRECT" if is_ground_truth_correct(pair) else "INCORRECT"
        )
    return kwargs


# ---------------------------------------------------------------------------
# Parquet building
# ---------------------------------------------------------------------------


def build_generator_feedback_parquet(
    pairs: Sequence[dict],
    output_path: Path,
    split: str,
) -> int:
    """Build a multi-turn GRPO training parquet from selected feedback pairs."""
    rows = []
    for pair in pairs:
        feedback = pair.get("verifier_feedback") or GENERIC_REPROMPT_FEEDBACK
        prompt = [
            {"role": "user", "content": pair["problem"]},
            {"role": "assistant", "content": pair["solution"]},
            {
                "role": "user",
                "content": REPROMPT_WITH_FEEDBACK.format(feedback=feedback),
            },
        ]
        extra_info = {
            "bin": pair["bin"],
            "problem_hash": pair["problem_hash"],
            "problem_idx": pair["problem_idx"],
            "chain_idx": pair["chain_idx"],
            "source_run": pair["source_run"],
            "true_reward_round0": pair["true_reward"],
            "verifier_verdict": pair["verifier_verdict"],
            "verifier_feedback": feedback,
            "verifier_prompt_id": pair.get(
                "verifier_prompt_id", VERIFY_UNCONDITIONAL_PROMPT_ID
            ),
            "verifier_feedback_leaked_answer": bool(
                pair.get(
                    "verifier_feedback_leaked_answer",
                    check_answer_leakage(feedback, pair.get("ground_truth_answer", "")),
                )
            ),
            "feedback_data_selection": pair.get(
                "feedback_data_selection", "verifier_incorrect"
            ),
            "original_problem": pair["problem"],
        }
        if split == "val":
            data_source = f"generator_feedback_{pair['bin']}"
        else:
            data_source = "generator_feedback"
        row = {
            "prompt": prompt,
            "data_source": data_source,
            "reward_model": {"style": "rule", "ground_truth": pair["ground_truth_answer"]},
            "extra_info": extra_info,
        }
        rows.append(row)

    table = pa.table(
        {
            "prompt": pa.array([_to_native(r["prompt"]) for r in rows]),
            "data_source": pa.array([r["data_source"] for r in rows]),
            "reward_model": pa.array([_to_native(r["reward_model"]) for r in rows]),
            "extra_info": pa.array([_to_native(r["extra_info"]) for r in rows]),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return len(rows)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_items(
    items: Sequence[dict],
    max_items: int,
    rng: random.Random,
    prefer_non_leaky_feedback: bool,
) -> List[dict]:
    """Sample items, optionally filling from non-leaky feedback first."""
    if len(items) <= max_items:
        return list(items)
    if not prefer_non_leaky_feedback:
        return rng.sample(list(items), max_items)

    non_leaky = [
        item for item in items
        if not item.get("verifier_feedback_leaked_answer", False)
    ]
    leaky = [
        item for item in items
        if item.get("verifier_feedback_leaked_answer", False)
    ]

    chosen: List[dict] = []
    if len(non_leaky) >= max_items:
        return rng.sample(non_leaky, max_items)
    chosen.extend(non_leaky)
    remaining = max_items - len(chosen)
    if remaining > 0:
        chosen.extend(rng.sample(leaky, min(remaining, len(leaky))))
    return chosen


def sample_prefer_non_leaky_per_problem(
    pairs: List[dict],
    max_per_problem: int,
    seed: int,
    prefer_non_leaky_feedback: bool,
) -> List[dict]:
    """Keep at most K rows per problem, optionally preferring non-leaky feedback."""
    if max_per_problem <= 0:
        return pairs

    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["bin"], pair["problem_hash"])].append(pair)

    rng = random.Random(seed)
    out: List[dict] = []
    for key in sorted(grouped.keys()):
        out.extend(
            sample_items(
                grouped[key],
                max_per_problem,
                rng,
                prefer_non_leaky_feedback,
            )
        )
    return out


def sample_with_false_accept_priority(
    verifier_incorrect: Sequence[dict],
    false_accepts: Sequence[dict],
    max_per_problem: int,
    seed: int,
    prefer_non_leaky_feedback: bool = False,
) -> List[dict]:
    """Sample up to K rows per problem, filling with false accepts only if needed."""
    if max_per_problem <= 0:
        return list(verifier_incorrect) + list(false_accepts)

    incorrect_by_problem: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    false_accept_by_problem: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for pair in verifier_incorrect:
        incorrect_by_problem[(pair["bin"], pair["problem_hash"])].append(pair)
    for pair in false_accepts:
        false_accept_by_problem[(pair["bin"], pair["problem_hash"])].append(pair)

    rng = random.Random(seed)
    selected: List[dict] = []
    keys = sorted(set(incorrect_by_problem) | set(false_accept_by_problem))
    for key in keys:
        incorrect_items = incorrect_by_problem.get(key, [])
        chosen_incorrect = sample_items(
            incorrect_items,
            max_per_problem,
            rng,
            prefer_non_leaky_feedback,
        )
        selected.extend(chosen_incorrect)

        remaining = max_per_problem - len(chosen_incorrect)
        if remaining <= 0:
            continue
        false_accept_items = false_accept_by_problem.get(key, [])
        selected.extend(
            sample_items(
                false_accept_items,
                remaining,
                rng,
                prefer_non_leaky_feedback,
            )
        )
    return selected


def is_ground_truth_correct(pair: dict) -> bool:
    """Return whether the round-0 solution matches the ground-truth answer."""
    return float(pair.get("true_reward", 0.0)) > 0.5


def select_feedback_pairs(
    verified: Sequence[dict],
    include_false_accepts_with_default_feedback: bool,
) -> Tuple[List[dict], List[dict], Counter]:
    """Split verified rollouts into normal rejects and optional false accepts."""
    verifier_incorrect: List[dict] = []
    false_accepts: List[dict] = []
    stats: Counter = Counter()

    for pair in verified:
        verdict = pair.get("verifier_verdict")
        gt_correct = is_ground_truth_correct(pair)

        if verdict == "INCORRECT":
            aug = dict(pair)
            if gt_correct:
                stats["false_reject"] += 1
                aug["feedback_data_selection"] = "false_reject"
            else:
                stats["true_reject"] += 1
                aug["feedback_data_selection"] = "true_reject"
            verifier_incorrect.append(aug)
            continue

        if verdict == "CORRECT" and not gt_correct:
            stats["false_accept"] += 1
            if include_false_accepts_with_default_feedback:
                aug = dict(pair)
                aug["verifier_feedback"] = None
                aug["feedback_data_selection"] = "false_accept_default_reprompt"
                false_accepts.append(aug)
            continue

        if verdict == "CORRECT" and gt_correct:
            stats["true_accept"] += 1
        else:
            stats["unparsed_or_other"] += 1

    return verifier_incorrect, false_accepts, stats


def ensure_leakage_annotations(pairs: Sequence[dict]) -> None:
    """Annotate loaded verifier cache rows with answer-leakage status in place."""
    for pair in pairs:
        if "verifier_feedback_leaked_answer" in pair:
            continue
        pair["verifier_feedback_leaked_answer"] = check_answer_leakage(
            pair.get("verifier_feedback") or "",
            pair.get("ground_truth_answer", ""),
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def print_verification_stats(pairs: Sequence[dict]):
    n_total = len(pairs)
    n_incorrect = sum(1 for p in pairs if p["verifier_verdict"] == "INCORRECT")
    n_correct = n_total - n_incorrect

    # Among verifier-INCORRECT, how many are GT correct (false rejections)?
    incorrect_pairs = [p for p in pairs if p["verifier_verdict"] == "INCORRECT"]
    n_false_reject = sum(1 for p in incorrect_pairs if float(p["true_reward"]) > 0.5)
    n_true_reject = len(incorrect_pairs) - n_false_reject

    print(f"  Verifier verdicts: CORRECT={n_correct}, INCORRECT={n_incorrect} "
          f"(of {n_total} total)")
    print(f"  Among INCORRECT: true_reject={n_true_reject}, "
          f"false_reject={n_false_reject}")

    by_bin = defaultdict(lambda: {"total": 0, "incorrect": 0})
    for p in pairs:
        by_bin[p["bin"]]["total"] += 1
        if p["verifier_verdict"] == "INCORRECT":
            by_bin[p["bin"]]["incorrect"] += 1
    for bin_name in sorted(by_bin):
        b = by_bin[bin_name]
        print(f"    {bin_name}: {b['incorrect']}/{b['total']} INCORRECT")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare offline generator-feedback training data"
    )
    p.add_argument(
        "--generator_model",
        type=str,
        default="checkpoints/generators/grpo_generator_standard_hard_Qwen-Qwen3-8B_lr1e-6/global_step_800/huggingface",
    )
    p.add_argument(
        "--verifier_model",
        type=str,
        default="checkpoints/merged_hf/opd_grpo_verifier_hard_Qwen-Qwen3-8B_alpha0.5_lr1e-6_opd1.0_pg1.0_k3_initrlgen_step2200",
    )
    p.add_argument(
        "--verifier_prompt_id",
        type=str,
        default=VERIFY_UNCONDITIONAL_PROMPT_ID,
        help="Prompt template used to generate feedback (strategy O = identify_mistake).",
    )
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
        "--max_per_problem",
        type=int,
        default=3,
        help=(
            "Keep at most this many selected rollouts per problem. With false "
            "accept inclusion enabled, verifier-INCORRECT rows are prioritized "
            "and false accepts only fill remaining slots. Set to 0 for no limit."
        ),
    )
    p.add_argument(
        "--include_false_accepts_with_default_feedback",
        action="store_true",
        help=(
            "Also add false accepts (ground-truth wrong, verifier says CORRECT) "
            "using the generic feedback reprompt. Output filenames get an _fa "
            "suffix to avoid overwriting the default verifier-INCORRECT data."
        ),
    )
    p.add_argument(
        "--prefer_non_leaky_feedback",
        action="store_true",
        help=(
            "When sampling multiple candidate feedback rows for the same problem, "
            "prefer rows whose feedback does not trigger the Experiment 0 answer "
            "leakage checker."
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
    random.seed(args.seed)
    bins = [x.strip() for x in args.bins.split(",") if x.strip()]

    requested_splits: List[str] = []
    if args.split in {"train", "both"}:
        requested_splits.append("train")
    if args.split in {"val", "both"}:
        requested_splits.append("val")

    if not requested_splits:
        print("Nothing to do.")
        return

    # Resolve DAPO parquet paths
    parquet_paths_by_split: Dict[str, List[Path]] = {}
    for split in requested_splits:
        paths = resolve_dapo_parquets(args.data_dir, bins, split=split)
        if not paths:
            raise RuntimeError(f"No DAPO {split} parquets resolved.")
        parquet_paths_by_split[split] = paths

    # ------------------------------------------------------------------
    # Stage 1: Generate rollouts from RL generator
    # ------------------------------------------------------------------
    raw_pairs_by_split: Dict[str, List[dict]] = {}

    llm, tokenizer, gen_params, _, _ = init_vllm_engine(
        model_name=args.generator_model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        rollout_n=args.rollout_n,
    )
    try:
        for split in requested_splits:
            print(f"\nGenerating {split.upper()} rollouts with RL generator...")
            raw_pairs_by_split[split] = collect_from_rollouts(
                model_name=args.generator_model,
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

    # ------------------------------------------------------------------
    # Stage 2: Run verifier on all rollouts
    # ------------------------------------------------------------------
    verified_pairs_by_split: Dict[str, List[dict]] = {}
    for split in requested_splits:
        print(f"\nRunning verifier on {split.upper()} rollouts...")
        verified_pairs_by_split[split] = run_verifier_on_pairs(
            verifier_model=args.verifier_model,
            pairs=raw_pairs_by_split[split],
            verifier_prompt_id=args.verifier_prompt_id,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
        )

    # ------------------------------------------------------------------
    # Stage 3: Filter and build parquets
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    for split in requested_splits:
        verified = verified_pairs_by_split[split]
        ensure_leakage_annotations(verified)
        print(f"\n{split.upper()} verification stats:")
        print_verification_stats(verified)

        verifier_incorrect, false_accepts, selection_stats = select_feedback_pairs(
            verified,
            include_false_accepts_with_default_feedback=(
                args.include_false_accepts_with_default_feedback
            ),
        )
        print(f"  Total INCORRECT rollouts: {len(verifier_incorrect)}")
        print(
            "  Selection stats: "
            f"true_reject={selection_stats['true_reject']}, "
            f"false_reject={selection_stats['false_reject']}, "
            f"false_accept={selection_stats['false_accept']}, "
            f"true_accept={selection_stats['true_accept']}, "
            f"unparsed_or_other={selection_stats['unparsed_or_other']}"
        )
        if args.include_false_accepts_with_default_feedback:
            print(
                "  Adding false accepts with default feedback: "
                f"{len(false_accepts)}"
            )

        # Sample per problem
        split_seed = args.seed + (0 if split == "train" else 1000003)
        if args.include_false_accepts_with_default_feedback:
            selected = sample_with_false_accept_priority(
                verifier_incorrect,
                false_accepts,
                args.max_per_problem,
                split_seed,
                prefer_non_leaky_feedback=args.prefer_non_leaky_feedback,
            )
            selected_stats = Counter(
                pair.get("feedback_data_selection", "verifier_incorrect")
                for pair in selected
            )
            selected_leaky = sum(
                1 for pair in selected
                if pair.get("verifier_feedback_leaked_answer", False)
            )
            print(
                f"  After priority sampling (max {args.max_per_problem}/problem): "
                f"{len(selected)}"
            )
            print(
                "  Selected rows: "
                f"true_reject={selected_stats['true_reject']}, "
                f"false_reject={selected_stats['false_reject']}, "
                f"false_accept_default_reprompt="
                f"{selected_stats['false_accept_default_reprompt']}"
            )
            print(
                f"  Selected answer-leaking feedback rows: {selected_leaky}/"
                f"{len(selected)}"
            )
        elif args.max_per_problem > 0:
            verifier_incorrect = sample_prefer_non_leaky_per_problem(
                verifier_incorrect,
                args.max_per_problem,
                split_seed,
                prefer_non_leaky_feedback=args.prefer_non_leaky_feedback,
            )
            selected_leaky = sum(
                1 for pair in verifier_incorrect
                if pair.get("verifier_feedback_leaked_answer", False)
            )
            print(
                f"  After sampling INCORRECT (max {args.max_per_problem}/problem): "
                f"{len(verifier_incorrect)}"
            )
            print(
                f"  Selected answer-leaking feedback rows: {selected_leaky}/"
                f"{len(verifier_incorrect)}"
            )
            selected = verifier_incorrect
        else:
            selected = verifier_incorrect

        # Write per-bin parquets
        suffix = f"_k{args.max_per_problem}" if args.max_per_problem > 0 else ""
        if args.include_false_accepts_with_default_feedback:
            suffix += "_fa"
        by_bin: Dict[str, List[dict]] = defaultdict(list)
        for p in selected:
            by_bin[p["bin"]].append(p)
        for bin_name in sorted(by_bin):
            out_path = output_dir / f"dapo_generator_feedback_{split}_{bin_name}{suffix}.parquet"
            n = build_generator_feedback_parquet(by_bin[bin_name], out_path, split=split)
            print(f"  Wrote {n} rows -> {out_path}")

    print("\nDone. Next: train with verifier/run_grpo_generator_feedback_hard_qwen3_8b.sh")


if __name__ == "__main__":
    main()
