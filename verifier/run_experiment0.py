#!/usr/bin/env python3
"""Verifier feedback scaffolding eval (strategy O, vLLM backend).

Runs the unconditioned verifier (strategy O, "identify_mistake") in a
verification-refinement loop: the generator produces a solution, the verifier
judges it (VERDICT + FEEDBACK), and if rejected the generator is reprompted with
the feedback. Measures verifier accuracy and pass@1 improvement over rounds.

Generation uses vLLM. The verifier is a local vLLM model, either the generator
itself (``--verifier_mode self``) or a separately loaded model
(``--verifier_mode separate``).

Usage:
    # Self-verifier (generator judges its own solutions)
    python verifier/run_experiment0.py --verifier_mode self --max_rounds 3

    # Separate local vLLM verifier
    python verifier/run_experiment0.py \
        --generator_model Qwen/Qwen3-8B \
        --verifier_mode separate --verifier_model /path/to/merged_hf_verifier \
        --strategies O --max_rounds 3
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    TimeoutException = Exception
    math_metric = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from examples.sdpo_trainer.math_reward import compute_score


from verifier.prompt_templates import get_oracle_template
from verifier.strategy_registry import STRATEGIES

# ---------------------------------------------------------------------------
# Reprompt templates
# ---------------------------------------------------------------------------

REPROMPT_WITH_FEEDBACK = (
    "A reviewer found issues with your solution:\n\n"
    "{feedback}\n\n"
    "Please solve the problem again, fixing the identified errors. "
    "Put your final answer inside \\boxed{{}}."
)

ANSWER_EXTRACTOR = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_native(obj):
    """Recursively convert numpy types to native Python."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def load_val_parquet(path: str) -> list[dict]:
    """Load a val parquet into a list of sample dicts."""
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
        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        samples.append({
            "prompt": _to_native(prompt),
            "data_source": str(row.get("data_source", "")),
            "reward_model": _to_native(reward_model),
            "extra_info": _to_native(extra_info),
        })
    return samples


def apply_chat_template(tokenizer, messages, max_prompt_length=None):
    """Apply chat template with enable_thinking=False for Qwen3."""
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=False, enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    if max_prompt_length is not None:
        token_ids = tokenizer.encode(text)
        if len(token_ids) > max_prompt_length:
            token_ids = token_ids[:max_prompt_length]
            text = tokenizer.decode(token_ids)
    return text


def get_problem_text(sample):
    """Extract the user's problem text from the chat-format prompt."""
    for msg in reversed(sample["prompt"]):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def score_response(sample, response):
    """Score a response against the ground truth."""
    gt = sample["reward_model"].get("ground_truth", "")
    return compute_score(
        data_source=sample["data_source"],
        solution_str=response,
        ground_truth=gt,
    )


def extract_answer(response):
    """Parse and normalize a model answer for majority voting."""
    global ANSWER_EXTRACTOR

    if not response:
        return None

    if ANSWER_EXTRACTOR is None and math_metric is not None:
        ANSWER_EXTRACTOR = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

    if ANSWER_EXTRACTOR is not None:
        extracted = None
        try:
            _, extracted = ANSWER_EXTRACTOR(["\\boxed{0}"], [response])
        except TimeoutException:
            extracted = None
        except Exception:
            extracted = None
        if extracted:
            parsed_predictions = extracted[1]
            if parsed_predictions:
                parsed = parsed_predictions[0]
                if parsed is not None:
                    parsed_str = str(parsed).strip()
                    if parsed_str:
                        return parsed_str

    # Fallback: try boxed content first, then a trailing number/fraction.
    m = re.search(r"\\boxed\{([^}]*)\}", response)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            return candidate
    matches = re.findall(r"(-?\d+(?:/\d+)?(?:\.\d+)?)", response)
    if matches:
        return matches[-1].strip()
    return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_oracle_response(text):
    """Parse VERDICT and FEEDBACK from binary oracle response."""
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


def _to_verifier_score(verdict):
    """Normalize a binary oracle verdict to a verifier score in [0, 1]."""
    return 1.0 if str(verdict).upper() == "CORRECT" else 0.0


def judgment_for_chain(judgments, chain_idx):
    """Extract the single verifier judgement for one chain."""
    verdict, feedback, call_usage = judgments[chain_idx]
    score = _to_verifier_score(verdict)
    usage = {
        "input_tokens": call_usage.get("input_tokens", 0),
        "output_tokens": call_usage.get("output_tokens", 0),
    }
    if score >= 1.0 or not feedback:
        feedback = None
    return score, feedback, usage


# ---------------------------------------------------------------------------
# Oracle backends
# ---------------------------------------------------------------------------

def judge_batch_vllm(llm, tokenizer, items, oracle_template,
                     max_oracle_tokens=512, parse_fn=None,
                     label="self-verifier"):
    """Judge a batch using the same vLLM model as its own oracle."""
    if parse_fn is None:
        parse_fn = parse_oracle_response
    results = {}
    if not items:
        return results

    prompts = []
    indices = []
    for item in items:
        idx, problem_text, solution = item[0], item[1], item[2]
        # The verifier prompt only ever receives the problem and the student's
        # solution -- never the ground-truth answer.
        oracle_text = oracle_template.format(problem=problem_text,
                                             solution=solution)
        messages = [{"role": "user", "content": oracle_text}]
        prompt_str = apply_chat_template(tokenizer, messages)
        prompts.append(prompt_str)
        indices.append(idx)

    gen_params = SamplingParams(temperature=0.3, top_p=1.0,
                                max_tokens=max_oracle_tokens, n=1)
    outputs = llm.generate(prompts, gen_params)

    for j, out in enumerate(outputs):
        idx = indices[j]
        response_text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        verdict, feedback = parse_fn(response_text)
        if verdict is None:
            verdict = "INCORRECT"
        usage = {"input_tokens": 0, "output_tokens": n_tokens}
        results[idx] = (verdict, feedback, usage)

    print(f"    Judged {len(items)}/{len(items)} ({label})", flush=True)
    return results


# ---------------------------------------------------------------------------
# Core feedback loop
# ---------------------------------------------------------------------------

def run_feedback_loop(problems, generate_fn, judge_fn,
                      prompt_id, reprompt_mode,
                      max_rounds=3, chains_per_problem=8):
    """Run the verifier feedback scaffolding loop.

    Uses last_turn reprompting: original problem + last solution + verifier
    feedback. No multi-turn history accumulation.

    Args:
        generate_fn: Callable taking list of message lists (chat format),
            returning list of (response_text, n_output_tokens) tuples.

    Returns (results, total_model_tokens, total_oracle_tokens).
    """
    oracle_template = get_oracle_template(prompt_id)

    # Expand: each problem x chains_per_problem
    n_chains = len(problems) * chains_per_problem
    expanded = []
    for p in problems:
        for _ in range(chains_per_problem):
            expanded.append(p)

    # Per-chain state
    chain_responses = [None] * n_chains
    chain_conversation = [None] * n_chains
    chain_tokens_model = [0] * n_chains
    chain_tokens_oracle = [0] * n_chains
    chain_oracle_calls = [0] * n_chains
    chain_model_calls = [0] * n_chains
    chain_done = [False] * n_chains
    chain_oracle_verdict = [None] * n_chains
    chain_round_history = [[] for _ in range(n_chains)]
    chain_score_per_round = [[] for _ in range(n_chains)]
    chain_parsed_answer_per_round = [[] for _ in range(n_chains)]
    chain_verifier_score_per_round = [[] for _ in range(n_chains)]

    # Round 0: initial generation
    messages_r0 = [p["prompt"] for p in expanded]
    print(f"  Round 0: generating {len(messages_r0)} initial responses...")
    gen_results_r0 = generate_fn(messages_r0)

    for i, (resp, n_tok) in enumerate(gen_results_r0):
        chain_responses[i] = resp
        chain_tokens_model[i] += n_tok
        chain_model_calls[i] += 1
        chain_conversation[i] = list(expanded[i]["prompt"]) + [
            {"role": "assistant", "content": resp},
        ]
        sc = score_response(expanded[i], resp)
        chain_score_per_round[i].append(sc)
        chain_parsed_answer_per_round[i].append(extract_answer(resp))

    # Feedback rounds
    for round_idx in range(max_rounds):
        active = [i for i in range(n_chains) if not chain_done[i]]
        if not active:
            print(f"  Round {round_idx + 1}: all chains done")
            break

        # Judge previous responses
        judge_items = []
        for i in active:
            judge_items.append(
                (i, get_problem_text(expanded[i]), chain_responses[i])
            )
        print(
            f"  Round {round_idx + 1}: judging {len(judge_items)} chains..."
        )
        judgments = judge_fn(judge_items, oracle_template)

        # Process judgments
        incorrect = []
        n_correct = 0
        judged_scores = {}

        for i in active:
            score, feedback, usage = judgment_for_chain(judgments, i)
            chain_oracle_calls[i] += 1
            chain_tokens_oracle[i] += usage["input_tokens"] + usage["output_tokens"]

            actual_verdict = "CORRECT" if score >= 1.0 - 1e-9 else "INCORRECT"

            chain_oracle_verdict[i] = actual_verdict
            round_event = {
                "round": round_idx + 1, "phase": "judge",
                "feedback": feedback,
            }
            round_event["verdict"] = actual_verdict
            judged_scores[i] = score
            round_event["verifier_score"] = score
            round_event["decision_source"] = "verifier"
            chain_round_history[i].append(round_event)
            if actual_verdict == "CORRECT":
                chain_done[i] = True
                n_correct += 1
            else:
                incorrect.append({
                    "idx": i,
                    "feedback": feedback,
                    "round_event": round_event,
                })

        # Record verifier scores for all chains
        for i in range(n_chains):
            if i in judged_scores:
                chain_verifier_score_per_round[i].append(judged_scores[i])
            elif chain_verifier_score_per_round[i]:
                chain_verifier_score_per_round[i].append(
                    chain_verifier_score_per_round[i][-1])

        print(f"    {n_correct} correct, {len(incorrect)} incorrect")
        if not incorrect:
            break

        # Skip regeneration on the last round — no subsequent judge to evaluate it
        if round_idx == max_rounds - 1:
            break

        # Construct reprompts (always last_turn for verifier experiments)
        reprompt_messages = []
        reprompt_indices = []

        for item in incorrect:
            i = item["idx"]
            feedback = item["feedback"]
            user_msg = REPROMPT_WITH_FEEDBACK.format(
                feedback=feedback or "Your solution appears to be incorrect.")

            msgs = list(expanded[i]["prompt"]) + [
                {"role": "assistant", "content": chain_responses[i]},
                {"role": "user", "content": user_msg},
            ]
            reprompt_messages.append(msgs)
            reprompt_indices.append(i)

        # Generate new responses
        print(f"  Round {round_idx + 1}: generating {len(reprompt_messages)} responses...")
        gen_results = generate_fn(reprompt_messages)

        for j, i in enumerate(reprompt_indices):
            resp, n_tok = gen_results[j]
            chain_responses[i] = resp
            chain_tokens_model[i] += n_tok
            chain_model_calls[i] += 1
            chain_conversation[i] = list(expanded[i]["prompt"]) + [
                {"role": "assistant", "content": resp},
            ]
            chain_round_history[i].append({
                "round": round_idx + 1, "phase": "regen",
            })

        # Score all chains
        for i in range(n_chains):
            sc = score_response(expanded[i], chain_responses[i])
            chain_score_per_round[i].append(sc)
            chain_parsed_answer_per_round[i].append(
                extract_answer(chain_responses[i])
            )

    # Compile results
    results = []
    idx = 0
    for prob_i, problem in enumerate(problems):
        chains = []
        for _ in range(chains_per_problem):
            final_score = score_response(problem, chain_responses[idx])
            parsed_answers = chain_parsed_answer_per_round[idx]
            chains.append({
                "response": chain_responses[idx],
                "score": final_score,
                "initial_score": chain_score_per_round[idx][0],
                "score_per_round": chain_score_per_round[idx],
                "parsed_answer": parsed_answers[-1] if parsed_answers else None,
                "parsed_answer_per_round": parsed_answers,
                "tokens_model": chain_tokens_model[idx],
                "tokens_oracle": chain_tokens_oracle[idx],
                "model_calls": chain_model_calls[idx],
                "oracle_calls": chain_oracle_calls[idx],
                "oracle_verdict": chain_oracle_verdict[idx],
                "rounds_used": sum(
                    1 for e in chain_round_history[idx]
                    if e.get("phase") == "judge"
                ),
                "round_history": chain_round_history[idx],
                "verifier_score_per_round": chain_verifier_score_per_round[idx],
            })
            idx += 1
        results.append({"problem_idx": prob_i, "chains": chains})

    return results, sum(chain_tokens_model), sum(chain_tokens_oracle)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results, strategy_name, bin_name, max_rounds):
    """Compute aggregate and per-round metrics."""
    n_problems = len(results)
    if n_problems == 0:
        return {}

    all_chains = [c for r in results for c in r["chains"]]
    n_chains = len(all_chains)
    n_cpp = len(results[0]["chains"])

    # Aggregate
    pass1_sum = 0
    pass_n_count = 0
    initial_pass1_sum = 0
    initial_pass_n_count = 0

    for r in results:
        scores = [c["score"] for c in r["chains"]]
        initial_scores = [c["initial_score"] for c in r["chains"]]
        pass1_sum += sum(1 for s in scores if s > 0.5) / len(scores)
        if any(s > 0.5 for s in scores):
            pass_n_count += 1
        initial_pass1_sum += sum(1 for s in initial_scores if s > 0.5) / len(initial_scores)
        if any(s > 0.5 for s in initial_scores):
            initial_pass_n_count += 1

    # Verifier accuracy
    oracle_agrees = oracle_judged = 0
    for c in all_chains:
        v = c["oracle_verdict"]
        if v is None:
            continue
        oracle_judged += 1
        if isinstance(v, (int, float)):
            oracle_correct = (v == 1.0)
        else:
            oracle_correct = (v == "CORRECT")
        if oracle_correct == (c["score"] > 0.5):
            oracle_agrees += 1

    # Per-round pass@1
    per_round = {}
    for rd in range(max_rounds + 1):
        rd_sum = 0
        for r in results:
            rd_scores = []
            for c in r["chains"]:
                spr = c["score_per_round"]
                rd_scores.append(spr[min(rd, len(spr) - 1)])
            rd_sum += sum(1 for s in rd_scores if s > 0.5) / len(rd_scores)
        per_round[f"pass@1_r{rd}"] = round(rd_sum / n_problems, 4)

    metrics = {
        "strategy": strategy_name,
        "bin": bin_name,
        "n_problems": n_problems,
        "n_chains": n_chains,
        "chains_per_problem": n_cpp,
        "initial_pass@1": round(initial_pass1_sum / n_problems, 4),
        "initial_pass@N": round(initial_pass_n_count / n_problems, 4),
        "pass@1": round(pass1_sum / n_problems, 4),
        "pass@N": round(pass_n_count / n_problems, 4),
        "improvement": round((pass1_sum - initial_pass1_sum) / n_problems, 4),
        "avg_rounds": round(sum(c["rounds_used"] for c in all_chains) / n_chains, 2),
        "avg_model_tokens": round(sum(c["tokens_model"] for c in all_chains) / n_chains, 1),
        "avg_oracle_tokens": round(sum(c["tokens_oracle"] for c in all_chains) / n_chains, 1),
        "verifier_accuracy": round(oracle_agrees / oracle_judged, 4) if oracle_judged > 0 else None,
    }
    metrics.update(per_round)

    # Verifier calibration (verifier score vs true reward)
    has_verifier = any(c.get("verifier_score_per_round") for c in all_chains)
    if has_verifier:
        for rd in range(max_rounds + 1):
            vs_sum = tr_sum = 0.0
            for c in all_chains:
                vspr = c.get("verifier_score_per_round", [])
                spr = c["score_per_round"]
                vs_sum += vspr[min(rd, len(vspr) - 1)] if vspr else 0.0
                tr_sum += spr[min(rd, len(spr) - 1)]
            metrics[f"mean_verifier_score_r{rd}"] = round(vs_sum / n_chains, 4)
            metrics[f"mean_true_reward_r{rd}"] = round(tr_sum / n_chains, 4)

        final_vs = []
        final_tr = []
        for c in all_chains:
            vspr = c.get("verifier_score_per_round", [])
            if vspr:
                final_vs.append(vspr[-1])
                final_tr.append(c["score"])
        if len(final_vs) >= 2:
            mean_vs = sum(final_vs) / len(final_vs)
            mean_tr = sum(final_tr) / len(final_tr)
            cov = sum((v - mean_vs) * (t - mean_tr) for v, t in zip(final_vs, final_tr))
            var_vs = sum((v - mean_vs) ** 2 for v in final_vs)
            var_tr = sum((t - mean_tr) ** 2 for t in final_tr)
            denom = (var_vs * var_tr) ** 0.5
            metrics["verifier_true_correlation"] = round(cov / denom if denom > 0 else 0.0, 4)

    return metrics


def _detail_key(detail):
    """Stable key for one results.jsonl entry."""
    return (detail.get("strategy"), detail.get("bin"), detail.get("problem_idx"))


def _prepare_detail_for_save(detail):
    """Copy and truncate large response payloads before writing JSONL."""
    saved = dict(detail)
    saved["chains"] = [dict(chain) for chain in detail["chains"]]
    for chain in saved["chains"]:
        if "response" in chain and len(chain["response"]) > 500:
            chain["response"] = chain["response"][:500] + "... [truncated]"
    return saved


def overwrite_results_jsonl(results_path, details):
    """Overwrite matching (strategy, bin, problem_idx) entries in results.jsonl."""
    results_path = Path(results_path)
    merged = {}
    unparsable_lines = []

    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    unparsable_lines.append(line)
                    continue
                merged[_detail_key(parsed)] = parsed

    for detail in details:
        saved = _prepare_detail_for_save(detail)
        merged[_detail_key(saved)] = saved

    tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for line in unparsable_lines:
            f.write(line + "\n")
        for detail in merged.values():
            f.write(json.dumps(detail) + "\n")
    tmp_path.replace(results_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Verifier feedback scaffolding eval (strategy O, vLLM)")
    p.add_argument("--generator_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--verifier_mode", type=str, default="self",
                   choices=["self", "separate"],
                   help="Use the generator itself or a separately loaded vLLM verifier.")
    p.add_argument(
        "--verifier_model",
        type=str,
        default=None,
        help=(
            "Verifier model name or path when --verifier_mode=separate. "
            "Can be an HF model name or merged model directory."
        ),
    )
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--max_prompt_length", type=int, default=8192)
    p.add_argument("--problems_per_bin", type=int, default=20)
    p.add_argument("--chains_per_problem", type=int, default=8)
    p.add_argument("--max_rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_files", type=str, default=None,
                   help="Comma-separated parquet paths (overrides --val_dir)")
    p.add_argument("--val_dir", type=str,
                   default="retrieval-analogy/data/val_parquets")
    p.add_argument("--output_dir", type=str,
                   default="verifier/output/experiment0")
    p.add_argument("--strategies", type=str, default="O",
                   help="Comma-separated strategies (O).")
    return p.parse_args()


def main():
    args = parse_args()
    import random
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse strategies
    strategy_keys = [s.strip().upper() for s in args.strategies.split(",")]
    for s in strategy_keys:
        if s not in STRATEGIES:
            print(f"Unknown strategy: {s}. Valid: {list(STRATEGIES.keys())}")
            return

    # Load problems
    all_problems = {}

    if args.val_files:
        # Load from explicit file paths (e.g. aime2024.parquet, hmmt2025.parquet)
        for fpath in args.val_files.split(","):
            fpath = fpath.strip()
            p = Path(fpath)
            if not p.exists():
                print(f"Warning: {p} not found, skipping")
                continue
            bin_name = p.stem  # e.g. "aime2025"
            samples = load_val_parquet(str(p))
            if len(samples) > args.problems_per_bin:
                samples = random.sample(samples, args.problems_per_bin)
            all_problems[bin_name] = samples
            print(f"Loaded {len(samples)} problems from {bin_name}")
    else:
        # Default: load DAPO difficulty bins from val_dir
        val_dir = Path(args.val_dir)
        bins = ["0.0_0.0", "0.0_0.2"]
        for bin_name in bins:
            parquet_path = val_dir / f"val_{bin_name}.parquet"
            if not parquet_path.exists():
                print(f"Warning: {parquet_path} not found, skipping")
                continue
            samples = load_val_parquet(str(parquet_path))

            if len(samples) > args.problems_per_bin:
                samples = random.sample(samples, args.problems_per_bin)
            all_problems[bin_name] = samples

            print(f"Loaded {len(samples)} problems from bin {bin_name}")

    if not all_problems:
        print("No problems loaded!")
        return

    if args.verifier_mode == "separate" and not args.verifier_model:
        print("Error: --verifier_model is required when --verifier_mode=separate")
        return

    def create_vllm_model(model_name, label):
        print(f"\nInitializing vLLM {label}: {model_name}")
        llm_kwargs = dict(
            model=model_name, trust_remote_code=True, dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
        )
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
        llm = LLM(**llm_kwargs)
        tokenizer = llm.get_tokenizer()
        return llm, tokenizer

    def make_vllm_state(model_name, label):
        return {
            "model_name": model_name,
            "label": label,
            "llm": None,
            "tokenizer": None,
        }

    def release_vllm_state(state):
        if state is None or state["llm"] is None:
            return
        print(
            f"\nReleasing vLLM {state['label']}: {state['model_name']}"
        )
        state["llm"].llm_engine.engine_core.shutdown()
        state["llm"] = None
        state["tokenizer"] = None

    def ensure_vllm_state_loaded(state, *peer_states):
        for peer_state in peer_states:
            if peer_state is not None and peer_state is not state:
                release_vllm_state(peer_state)
        if state["llm"] is None:
            state["llm"], state["tokenizer"] = create_vllm_model(
                state["model_name"], state["label"]
            )
        return state["llm"], state["tokenizer"]

    generator_vllm_state = None
    verifier_vllm_state = None

    # Initialize vLLM generation backend
    if LLM is None:
        print("Error: vLLM is required. Install with: pip install vllm")
        return

    generator_vllm_state = make_vllm_state(args.generator_model, "generator")

    max_prompt_length = args.max_prompt_length

    def generate_fn(messages_list):
        llm, tokenizer = ensure_vllm_state_loaded(
            generator_vllm_state, verifier_vllm_state
        )
        gen_params = SamplingParams(temperature=1.0, top_p=1.0,
                                    max_tokens=4096, n=1)
        prompts = [apply_chat_template(tokenizer, m, max_prompt_length)
                   for m in messages_list]
        outputs = llm.generate(prompts, gen_params)
        return [(out.outputs[0].text, len(out.outputs[0].token_ids))
                for out in outputs]

    # Initialize verifier backend (judging)
    judge_fn = None
    if args.verifier_mode == "self":
        print("Verifier: self (same vLLM model)")
        verifier_vllm_state = generator_vllm_state

        def judge_fn(items, oracle_template, **kwargs):
            llm, tokenizer = ensure_vllm_state_loaded(verifier_vllm_state)
            return judge_batch_vllm(llm, tokenizer, items,
                                    oracle_template, **kwargs)
    else:
        # Separate vLLM verifier model
        same_vllm_model = args.verifier_model == args.generator_model
        if same_vllm_model:
            verifier_vllm_state = generator_vllm_state
        else:
            verifier_vllm_state = make_vllm_state(
                args.verifier_model, "verifier"
            )
        print(f"Verifier: separate vLLM model ({args.verifier_model})")

        def judge_fn(items, oracle_template, **kwargs):
            ver_llm, ver_tokenizer = ensure_vllm_state_loaded(
                verifier_vllm_state, generator_vllm_state
            )
            return judge_batch_vllm(ver_llm, ver_tokenizer, items,
                                    oracle_template, **kwargs)

    # Run strategies
    all_metrics = []
    all_detailed = []

    for strat_key in strategy_keys:
        strat = STRATEGIES[strat_key]
        print(f"\n{'='*70}")
        print(f"Strategy {strat_key}: {strat['name']}")
        print(f"{'='*70}")

        for bin_name, problems in all_problems.items():
            print(f"\n--- Bin {bin_name} ({len(problems)} problems) ---")
            t0 = time.time()

            results, model_tok, oracle_tok = run_feedback_loop(
                problems=problems, generate_fn=generate_fn,
                judge_fn=judge_fn,
                prompt_id=strat.get("prompt_id", strat["feedback_type"]),
                reprompt_mode=strat["reprompt_mode"],
                max_rounds=args.max_rounds,
                chains_per_problem=args.chains_per_problem,
            )

            elapsed = time.time() - t0
            metrics = compute_metrics(
                results, f"{strat_key}: {strat['name']}", bin_name,
                args.max_rounds)
            metrics["elapsed_s"] = round(elapsed, 1)
            all_metrics.append(metrics)

            for r in results:
                problem = problems[r["problem_idx"]]
                all_detailed.append({
                    "strategy": f"{strat_key}: {strat['name']}",
                    "bin": bin_name,
                    "problem_idx": r["problem_idx"],
                    "ground_truth": problem["reward_model"].get("ground_truth", ""),
                    "chains": r["chains"],
                })

            cpp = metrics.get("chains_per_problem", 1)
            verifier_acc = metrics.get("verifier_accuracy")
            verifier_acc_str = f"{verifier_acc:.3f}" if verifier_acc is not None else "n/a"
            print(
                f"  init_p@1={metrics['initial_pass@1']:.4f}  "
                f"p@1={metrics['pass@1']:.4f}  "
                f"(+{metrics['improvement']:+.4f})  "
                f"p@{cpp}={metrics['pass@N']:.4f}  "
                f"verifier_acc={verifier_acc_str}  "
                f"time={elapsed:.1f}s"
            )

            for rd in range(args.max_rounds + 1):
                line = f"    R{rd}: p@1={metrics[f'pass@1_r{rd}']:.4f}"
                vs_key = f"mean_verifier_score_r{rd}"
                if vs_key in metrics:
                    line += f"  verifier={metrics[vs_key]:.4f}  true={metrics[f'mean_true_reward_r{rd}']:.4f}"
                print(line)

    # Save results
    results_path = output_dir / "results.jsonl"
    overwrite_results_jsonl(results_path, all_detailed)

    # Final comparison table
    print(f"\n{'='*100}")
    print("Verifier feedback scaffolding results")
    print(f"{'='*100}")

    for bin_name in all_problems:
        print(f"\nBin: {bin_name}")
        header = (f"  {'Strategy':<40} {'init p@1':>9} {'p@1':>7} "
                  f"{'delta':>7} {'p@N':>7} {'V-Acc':>7}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for m in all_metrics:
            if m["bin"] != bin_name:
                continue
            acc = m.get("verifier_accuracy")
            acc_str = f"{acc:>7.3f}" if acc is not None else "    N/A"
            print(
                f"  {m['strategy']:<40} "
                f"{m['initial_pass@1']:>9.4f} "
                f"{m['pass@1']:>7.4f} "
                f"{m['improvement']:>+7.4f} "
                f"{m['pass@N']:>7.4f} "
                f"{acc_str}"
            )

    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
