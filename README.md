# Agentic Verifier

This repository contains the code and data interfaces for reproducing experiments on agentic multi-turn verification, including Best-of-N (BoN) verification, iterative refinement, and verifier training.

## Setup

```bash
pip install -r requirements.txt
```

## Prepare Your Data

All inference datasets are stored in JSONL format, and train datasets should be in parquet format.

### Evaluation Data

We support BoN evaluation and Refine evaluation.

#### BoN Evaluation Format
```json
{
  "idx": 0,
  "input": "<prompt text for the candidate model>",
  "question": "<raw question>",
  "answer": "optional standard answer trace",
  "ground_truth": "(3,\\frac{\\pi}{2})",
  "samples": [
    "candidate answer 0",
    "candidate answer 1"
  ],
  "evaluations": [
    {
      "correct": true,
      "parsed_gt": "(3,\\frac{\\pi}{2})",
      "parsed_pred": "(3,\\frac{\\pi}{2})",
      "mathd_equal": true,
      "sympy_equal": false,
      "sampling_id": 0
    }
  ]
}
```

#### Alignment rule

* samples and evaluations are one-to-one aligned by index.
* Each evaluation corresponds to exactly one candidate answer.
* This alignment is required for correct BoN aggregation and metric computation.

#### Refine Evaluation Format

```json

{
  "idx": 0,
  "input": "<prompt text for the candidate model>",
  "question": "<raw question>",
  "answer": "optional standard answer trace",
  "ground_truth": "33",
  "refine_rounds": [
    {
      "answer": "<candidate reasoning trace>",
      "cand_correct": false,
      "cand_grade": {
        "parsed_gt": "33",
        "parsed_pred": "13",
        "correct": false,
        "mathd_equal": false,
        "sympy_equal": false
      }
    }
  ]
}
```

If no candidate answer is provided, one will be generated in the first refinement round.

### Training Data

#### Train Data Format(RL)

```json
{
  "idx": 708,
  "data_source": "rm_bool",
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "reward_model": {
    "ground_truth": true,
    "style": "rule"
  },
  "extra_info": {
    "meta": {},
    "problem": "<original question>",
    "solution": "<candidate solution>"
  }
}
```

#### Notes

* data_source must be "rm_bool".
* The training input is constructed from extra_info.problem + extra_info.solution.
* prompt is kept only for verl framework checking and is not used for training.

## Run Inference
### BoN Verification


```bash
bash examples/run_verify.sh
```

or
```bash
bash examples/run_verify_entry.sh \
  --task-name <task> \
  --exp-name <exp> \
  --model-path <model> \
  --input <input.jsonl> \
  --output-dir <output_dir> \
  --num-workers <N> \
  --enable-thinking
```

#### BoN-specific parameters

* num-workers: number of parallel verifier workers
* enable-thinking: enables model-specific reasoning mode
* start-idx / append: allow partial or resumed evaluation

BoN performs single-pass verification over a fixed candidate set and does not involve iterative interaction.

### Refine Verification

```bash
bash examples/run_refine.sh
```

or

```bash
bash examples/run_refine_entry.sh \
  --candidate-config config/refine_candidate.yaml \
  --verifier-config  config/refine_verifier.yaml \
  --input <input.jsonl> \
  --output <final.jsonl> \
  --round-output-dir <round_dir> \
  --metrics-output-dir <metrics_dir> \
  --exp-name <exp> \
  --verifier-type forward \
  --candidate-model-path <candidate_model> \
  --verifier-model-path <verifier_model> \
  --num-candidate-workers <Nc> \
  --num-verifier-workers <Nv> \
  --max-refine-rounds <K>
```

#### Refine-specific parameters

* num-candidate-workers: parallel candidate generation workers
* num-verifier-workers: parallel verifier workers
* max-refine-rounds: maximum verification–correction iterations
* verifier-type: multihead, forward, backward, or vanilla

Refine performs multi-round interaction, where candidates may be revised based on verifier feedback.

## Run Training

### Supervised Fine-Tuning (SFT)

SFT takes standard VERL multiturn train data.

```bash
bash examples/train_sft_multiturn.sh
```

### Reinforcement Learning (GRPO)

GRPO takes data of the specified format.

```bash
bash examples/train_grpo.sh
```

## Multi-Node / Multi-GPU Execution

- Both BoN inference and GRPO training support multi-node, multi-GPU execution.
- All machines should mount the same shared file system.
- Launched the same srcipts independently on each machine with multinode parameters (nnode > 1).
- One process automatically becomes the Ray head node; others join as workers.No manual cluster configuration is required.

