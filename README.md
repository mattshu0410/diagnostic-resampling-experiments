# Diagnostic Commitment in LLM Clinical Reasoning

## Overview

This repository contains the code and intermediate data to reproduce the main analyses in our paper. Our analysis generates empirical distributions of model diagnoses at various reasoning depths in chain-of-thought by resampling continuations at sentence boundaries and measuring changes in total variation distance.

Eight open-source reasoning models were run over free-response diagnostic vignettes. For each selected case a model contributes two base traces, one that reached the correct diagnosis and one that did not, and every sentence boundary of both was resampled 20 times. Our analysis asks the following: a) where in a trace does the answer distribution largely stabilise, how quickly it gets there, which sentences disproportionately collapse the answer distribution, and what the model goes on to emit after diagnostic commitment.

## What is included

Everything needed to reproduce the figures and tables in the paper.

* the resampling and analysis code
* the model, cohort, experiment and prompt configurations
* per-boundary answers and judged diagnosis entities for all 1,851,780 resampled continuations
* the intermediate CSVs the analysis notebooks read

Not included:

* **Raw rollout text** (~8 GB). Only the extracted answers and their judged entities are kept,
  which is what every analysis reads.
* **The `full_response` field of each base trace.** It contains the rendered prompt, and so the
  case text. Removed; nothing in the analysis reads it. `dctax score` and `dctax importance`
  still run against the stripped archive, `dctax select` and `dctax resample --stage sweep`
  do not.
* **NEJM CPC case text.** See [Licences and data restrictions](#licences-and-data-restrictions).
* **Model responses and screening runs.** Regenerating them needs GPUs; see [Compute](#compute).

## Repository structure

```
configs/
  cohorts/      the case collection and its source datasets
  models/       one file per model: checkpoint, decoding, trace format, answer readers
  prompts/      the diagnosis prompt and the three judge prompts
  resampling/   which models, pool size, rollout counts
src/dctax/
  config.py     loads the YAML and asserts it against the checkpoint
  data/         assembles the cohort from its source datasets
  generate/     chain-of-thought generation and diagnosis grading
  rollouts/     boundary location, resampling, scoring, answer grouping, importance
  utils/        trace extraction, sentence spans, answer readers, shared LLM client
analysis/
  1_convergence/       where in a trace the answer distribution settles
  2_sentence_effects/  which sentences move it, and what follows commitment
datasets/       MedQA and MedMCQA, filtered to free-response diagnostic cases
artifacts/
  rollouts/     per-run answers and traces, and the shared judgements
```

## Quickstart

Re-running the rollouts takes significant compute. The published artifacts let the analysis be
reproduced without doing so.

### Install

```bash
uv sync --extra analysis          # CPU only: reproduces every figure and table
uv sync --extra llm               # adds the LLM judge, for grading and answer grouping
uv sync --extra vllm              # adds generation and resampling, needs a GPU
```

### Reproducing the main results

The intermediate CSVs are already under `analysis/*/out/`, so the notebooks run directly:

```bash
jupyter lab analysis/1_convergence/convergence_tests.ipynb        # commitment depth, Table 1
jupyter lab analysis/2_sentence_effects/sentence_effects_tests.ipynb  # Figures 4 and 5
```

To regenerate those CSVs from the published artifacts instead:

```bash
python analysis/1_convergence/convergence.py       --config screen1000
python analysis/1_convergence/tvd.py               --config screen1000
python analysis/1_convergence/delta_sweep.py       --config screen1000
python analysis/2_sentence_effects/diagnosis_mentions.py --config screen1000
python analysis/2_sentence_effects/tvd_drop.py     --config screen1000
```

Artifacts are read from `artifacts/` by default; set `DCTAX_DATA_ROOT` to read them elsewhere.

## Data

### Case datasets

The cohort is 2,073 free-response diagnostic vignettes.

| source | cohort | screening pool | selected for the sweep |
| --- | ---: | ---: | ---: |
| MedQA (USMLE board questions) | 512 | 349 | 33 |
| MedMCQA (AIIMS and NEET board questions) | 1,259 | 349 | 39 |
| NEJM CPC (Case Records of the MGH) | 302 | 302 | 29 |
| **total** | **2,073** | **1,000** | **101** |

The pool is drawn balanced across sources rather than proportionally, which is why NEJM CPC is
taken in full. Multiple-choice options are removed and non-diagnostic questions filtered out.

### Provided artifacts

Under `artifacts/rollouts/`:

| file | what it holds |
| --- | --- |
| `pool_screen1000.json` | the 1,000 screened cases and their source |
| `cases_screen1000.json` | the 101 selected cases and which model sweeps each |
| `grades_screen1000.parquet` | 135,392 answer strings, their judged entity, and correctness |
| `screen1000-<model>/traces.jsonl` | 1,213 base traces: sentences and their token spans |
| `screen1000-<model>/rollout_scores.parquet` | the answer of every resampled continuation |

## Pipeline

Stages run in the order below. Each reads the experiment configuration from
`configs/resampling/screen1000.yaml`.

### 1. Generation

```bash
dctax cohort   --cohort main2073
dctax generate --model <slug> --cohort main2073
```

Each model is prompted through its own chat template with the decoding settings from its
checkpoint's `generation_config.json`, recorded per model under `configs/models/`. Responses are
stored with special tokens intact. Reasoning is extracted by `utils/traces.py`, which dispatches
on the model's declared envelope, and split into sentences with medspaCy.

### 2. Screening

```bash
dctax pool           --config screen1000
dctax resample       --config screen1000 --model <slug> --stage screen
dctax score          --config screen1000 --model <slug> --stage screen
dctax grade-rollouts --config screen1000 --stage screen
```

Eight continuations are generated from boundary 0 of each pooled case, which is the case
attempted with no reasoning in the prefix. `score` reads the final diagnosis from each
continuation; `grade-rollouts` groups the answers into clinical entities and scores one
representative per entity against the gold diagnosis.

### 3. Selection

```bash
dctax select --config screen1000
```

A case qualifies for a model when its screening samples include both a correct and an incorrect
answer, so the case can contribute a matched pair. Cases qualifying for at least six models form
a shared core swept by every model that qualifies; a model short of the floor is topped up from
its own qualifying list. The result is deterministic given the screening data.

### 4. Resampling

```bash
dctax resample --config screen1000 --model <slug> --stage sweep
dctax score    --config screen1000 --model <slug> --stage sweep
```

Each selected screening sample is rebuilt into a full trace, and 20 continuations are generated
from every one of its sentence boundaries. Prompts are token-id slices of the stored sequence,
so no text is re-rendered or re-encoded, and decoding comes from the model's own configuration.

### 5. Grading

```bash
dctax grade-rollouts --config screen1000 --stage sweep
```

Answers are grouped before they are scored, so two phrasings of one diagnosis cannot fall either
side of the correctness threshold. Grouping pools every model's answers for a case, so an entity
means the same thing in each model's distribution.

### 6. Analysis

See [Reproducing the main results](#reproducing-the-main-results).

## Compute

Resampling took approximately 420 H200 GPU hours. Grading and answer grouping are LLM judge
calls against a hosted API. All analysis is CPU-only and runs in minutes from the published
artifacts.

## Licences and data restrictions

**NEJM CPC.** The Case Records of the Massachusetts General Hospital are copyright the New
England Journal of Medicine and are not redistributed here. The 302 case ids are recorded in
`artifacts/rollouts/pool_screen1000.json`; obtain the cases separately and place them at the
path named in `configs/cohorts/main2073.yaml` to rebuild the cohort. The published artifacts
carry no case text from any source: only the models' own reasoning sentences and their answers.

**MedQA and MedMCQA.** Redistributed under `datasets/` in the filtered form used here. See the
original releases for their terms.
