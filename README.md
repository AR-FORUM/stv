# Self-Trained Verification 

Reference implementation for the paper:<br>
**[Self-Trained Verification for Training- and Test-Time Self-Improvement](https://arxiv.org/abs/2605.30290)**<br>
Chen Henry Wu, Aditi Raghunathan<br>
Project page: https://ar-forum.github.io/stv-webpage/
## Install

`verl/` is a customized fork of [verl](https://github.com/volcengine/verl); set it up
the same way you would set up upstream verl (PyTorch with a CUDA build, vLLM,
`flash-attn`, and `pip install -r requirements.txt`). Training assumes a single
8×GPU node.

## Data

The pipeline starts from **DAPO math rollouts** binned by the base generator's (post-trained Qwen-38B) pass@1. Point `--data_dir` at a directory of parquet files named
`{split}_{bin}.parquet` (default bins `0.0_0.0` = *Hardest*, `0.0_0.2` = *Hard*),
each row having:

- `prompt`: `list[{"role": ..., "content": ...}]` 
- `reward_model`: `{"ground_truth": "<answer>"}`
- `data_source`: the bin identifier

The data-prep scripts write their outputs to `verifier/data/` by default. The
distill step (`prepare_verifier_distill_data.py`) reads oracle reference solutions
for the reference-conditioned teacher from `--train_problems_dir` /
`--val_problems_dir` as `{split}_problems_{bin}.jsonl` with a `problem` field and an
oracle-solution field.

## Pipeline

All commands are run from this directory (the repo root), so that
`from examples...` / `from verifier...` imports resolve.

### Stage 1 — Train the STV verifier

```bash
# (a) DAPO rollouts -> RL verifier parquets
python verifier/prepare_verifier_data.py \
    --data_dir <your_dapo_rollouts> --output_dir verifier/data \
    --bins 0.0_0.0,0.0_0.2 --rollout_n 8

# (b) add the reference-conditioned teacher prompt (ref_prompt column)
python verifier/prepare_verifier_distill_data.py \
    --rl_train_file verifier/data/dapo_rl_train.parquet \
    --rl_val_file   verifier/data/dapo_rl_val.parquet

# (c) hybrid OPD + RL verifier training
TRAIN_FILES="['verifier/data/dapo_distill_train.parquet']" \
VAL_FILES="['verifier/data/dapo_rl_val.parquet']" \
    ./verifier/run_opd_rl_verifier_hard.sh
```

The student verifier sees `problem + solution`; the teacher reads a
privileged `ref_prompt` (problem + solution + verdict + reference solution). The
objective is `OPD_COEF * opd_loss + PG_COEF * policy_gradient_loss`.

### Verifier-in-the-loop generator training

```bash
# (a) build multi-turn [problem, attempt, verifier-feedback] prompts
python verifier/prepare_generator_feedback_data.py \
    --data_dir <your_dapo_rollouts> --output_dir verifier/data \
    --verifier_model <STV_verifier_checkpoint> --rollout_n 8

# (b) continue training the converged RLVR generator inside the V-R loop
TRAIN_FILES="['verifier/data/dapo_generator_feedback_train.parquet']" \
VAL_FILES="['verifier/data/dapo_generator_feedback_val.parquet']" \
    ./verifier/run_grpo_generator_feedback_hard_qwen3_8b.sh
```

### Verification-refinement evaluation

```bash
python verifier/run_experiment0.py \
    --generator_model <generator_checkpoint> \
    --verifier_mode self \
    --max_rounds 20 --problems_per_bin 150 --chains_per_problem 32
```

## Citation

```bibtex
@article{wu2026stv,
  title   = {Self-Trained Verification for Training- and Test-Time Self-Improvement},
  author  = {Wu, Chen Henry and Raghunathan, Aditi},
  journal = {arXiv preprint arXiv:2605.30290},
  year    = {2026},
}
```
