# Enhancing App Review Feature Extraction through Fine-Tuned Generative LLMs and Multi-Node Workflow Design

## Overview

This project enhances app review feature extraction using fine-tuned generative language models and multi-agent workflow design.

## GPT Workflow

### Step 1. Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

### Step 2. Data Preparation

Use the dataset provided by [T-FREX](https://github.com/nlp4se/t-frex/tree/main/data/T-FREX). The dataset contains in-domain and out-of-domain categories, each with 10 subcategories:
- in-domain: bin0...bin9
- out-of-domain: "COMMUNICATION", "HEALTH_AND_FITNESS", "LIFESTYLE", "MAPS_AND_NAVIGATION", "PERSONALIZATION", "PRODUCTIVITY", "SOCIAL", "TOOLS", "TRAVEL_AND_LOCAL", "WEATHER"

Run `make_dir.py` to create the required directory structure under `./data`:
```
mkdir data
python code/utils/make_dir.py
```

Place `test-set.txt` and `train-set.txt` in the `original_data` directory (see [T-FREX](https://github.com/nlp4se/t-frex/tree/main/data/T-FREX) for original data).

Run `conll_to_json_converter.py` to convert CoNLL format data to JSON format (saved to `formatted_original_data`):
```
python code/utils/conll_to_json_converter.py
```

### Step 3. Fine-tuning

Prepare GPT fine-tuning input data:
```
python code/fine-tuning/gpt/generate_finetuning_input_data.py
```

Run fine-tuning (`-f`: training file path, `-m`: base model name):
```
python code/fine-tuning/gpt/exec_finetuning.py -f "./data/in-domain/bin0/fine_tuning_data/train-set-finetuning-promptft1.json" -m "gpt-4.1-nano-2025-04-14"
```

### Step 4. Inference (Feature Extraction)

Single-node extraction:
```
python code/single-node/gpt/extraction.py -d ./data/in-domain/bin0/ -m "ft:gpt-4.1-nano-2025-04-14:personal::ABCDEFGH"
```

Multi-node extraction:
```
python code/multi-node/gpt/sp1_extraction.py -d "./data/in-domain/bin0/" -m "ft:gpt-4.1-nano-2025-04-14:personal::ABCDEFGH"
```

Extraction results are saved in the `feature_extracted_data` directory.

### Step 5. Evaluation

Single-node evaluation:
```
python code/single-node/evaluation.py -d "./data/in-domain/bin0/" -f "test-set-gpt-4-1-nano-2025-04-14-singleagent.json"
```

Multi-node evaluation:
```
python code/multi-node/gpt/sp1_evaluation.py -d "./data/in-domain/bin0/" -f "./data/in-domain/bin0/feature_extracted_data/test-set-ABCDEFGH-eval-opti.json"
```

Evaluation results are saved in the `evaluation_result` directory.

## OSS LLM Workflow (Qwen/GPT-OSS/Gemma/Llama on GPU server)

### Step 1. Setup

SSH into the shared GPU server (e.g. H200), move to this repository, and pull latest changes if needed:
```
git pull
```

Create and activate OSS environment:
```
python -m venv .venv-oss
source .venv-oss/bin/activate
```

Install dependencies using one of the two files:

- Base dependencies (flexible for day-to-day development):
```
pip install -r code/requirements-oss.txt
```

- Locked dependencies (reproducible runs on the same platform/GPU stack):
```
pip install -r code/requirements-oss-lock.txt
```

When the OSS environment is updated and validated, refresh the lock file:
```
python -m pip freeze > code/requirements-oss-lock.txt
```

### One-command pipeline (prepare -> train -> infer -> evaluate)

Run the full OSS flow in one command:
```
bash code/run_oss_pipeline.sh -m llama-3.1-8b -d in-domain/bin0
```

Run all in-domain bins (`bin0` to `bin9`) in one command:
```
bash code/run_oss_pipeline.sh -m llama-3.1-8b -d in-domain
```

Arguments:
- `--model` / `-m`: `qwen2.5-32b`, `qwen2.5-7b`, `gptoss` (or `gpt-oss`), `gemma-3-27b`, `gemma-3-4b`, `llama-3.1-8b`, or `llama-3.1-70b`
- `--dataset` / `-d`:
  - single dataset path (e.g. `in-domain/bin0`, `./data/in-domain/bin0`)
  - directory containing bins (e.g. `in-domain`, `./data/in-domain`) to run all `bin*` datasets in order

Legacy aliases `qwen`, `gemma`, and `llama` are still accepted and normalize to `qwen2.5-32b`, `gemma-3-27b`, and `llama-3.1-8b`.

This script executes:
1. `prepare_sft_dataset.py`
2. model-specific LoRA fine-tuning
3. model-specific extraction with the generated adapter
4. `evaluation.py`

Logs are saved to:
- `logs/train/<model>/<run_id>.log`
- `logs/infer/<model>/<run_id>.log`
- `logs/eval/<model>/<run_id>.log`

### Step 2. Data Preparation

Prepare SFT dataset:
```
python code/fine-tuning/oss_shared/prepare_sft_dataset.py -d ./data/in-domain/bin0/
```
This generates:
- `fine_tuning_data/train-set-sft-promptft1.jsonl` (training)
- `fine_tuning_data/validation-set-sft-promptft1.jsonl` (validation split from train-set, default 90/10)

### Step 3. Fine-tuning (LoRA)

Qwen fine-tuning:
```
RUN_ID="$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"
mkdir -p logs/train/qwen2.5-32b logs/pid/qwen2.5-32b
nohup python code/fine-tuning/qwen2.5-32b/train.py -d ./data/in-domain/bin0/ --run-id "${RUN_ID}" > logs/train/qwen2.5-32b/${RUN_ID}.log 2>&1 & echo $! > logs/pid/qwen2.5-32b/${RUN_ID}.pid
```

Gemma fine-tuning:
```
RUN_ID="$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"
mkdir -p logs/train/gemma-3-27b logs/pid/gemma-3-27b
nohup python code/fine-tuning/gemma-3-27b/train.py -d ./data/in-domain/bin0/ --run-id "${RUN_ID}" > logs/train/gemma-3-27b/${RUN_ID}.log 2>&1 & echo $! > logs/pid/gemma-3-27b/${RUN_ID}.pid
```

Llama fine-tuning:
```
RUN_ID="$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"
mkdir -p logs/train/llama-3.1-8b logs/pid/llama-3.1-8b
nohup python code/fine-tuning/llama-3.1-8b/train.py -d ./data/in-domain/bin0/ --run-id "${RUN_ID}" > logs/train/llama-3.1-8b/${RUN_ID}.log 2>&1 & echo $! > logs/pid/llama-3.1-8b/${RUN_ID}.pid
```

### Step 4. Inference

Qwen extraction:
```
TRAIN_RUN_ID="<run_id_from_training>"
INFER_RUN_ID="$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"
mkdir -p logs/infer/qwen2.5-32b logs/pid/qwen2.5-32b
nohup python code/single-node/qwen2.5-32b/extraction.py \
  -d ./data/in-domain/bin0/ \
  --adapter_dir artifacts/lora/qwen2.5-32b/bin0/${TRAIN_RUN_ID}/final \
  > logs/infer/qwen2.5-32b/${INFER_RUN_ID}.log 2>&1 & echo $! > logs/pid/qwen2.5-32b/${INFER_RUN_ID}.pid
```

Gemma extraction:
```
TRAIN_RUN_ID="<run_id_from_training>"
INFER_RUN_ID="$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"
mkdir -p logs/infer/gemma-3-27b logs/pid/gemma-3-27b
nohup python code/single-node/gemma-3-27b/extraction.py \
  -d ./data/in-domain/bin0/ \
  --adapter_dir artifacts/lora/gemma-3-27b/bin0/${TRAIN_RUN_ID}/final \
  > logs/infer/gemma-3-27b/${INFER_RUN_ID}.log 2>&1 & echo $! > logs/pid/gemma-3-27b/${INFER_RUN_ID}.pid
```

Llama extraction:
```
TRAIN_RUN_ID="<run_id_from_training>"
INFER_RUN_ID="$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"
mkdir -p logs/infer/llama-3.1-8b logs/pid/llama-3.1-8b
nohup python code/single-node/llama-3.1-8b/extraction.py \
  -d ./data/in-domain/bin0/ \
  --adapter_dir artifacts/lora/llama-3.1-8b/bin0/${TRAIN_RUN_ID}/final \
  > logs/infer/llama-3.1-8b/${INFER_RUN_ID}.log 2>&1 & echo $! > logs/pid/llama-3.1-8b/${INFER_RUN_ID}.pid
```

Extraction results are saved in the `feature_extracted_data` directory using run-id-based names:
- `test-set-qwen2-5-32b-singleagent-<run_id>.json`
- `test-set-gptoss-singleagent-<run_id>.json`
- `test-set-gemma-3-27b-singleagent-<run_id>.json`
- `test-set-llama-3-1-8b-singleagent-<run_id>.json`

### Step 5. Evaluation

Evaluate using existing GPT evaluation script:
```
python code/single-node/evaluation.py -d ./data/in-domain/bin0/ -f test-set-qwen2-5-32b-singleagent-<run_id>.json
```

Evaluation results are saved in `evaluation_result` with run-id-based names:
- `test-set-qwen2-5-32b-singleagent-<run_id>-result.txt`
- `test-set-gptoss-singleagent-<run_id>-result.txt`
- `test-set-gemma-3-27b-singleagent-<run_id>-result.txt`
- `test-set-llama-3-1-8b-singleagent-<run_id>-result.txt`

### Step 6. Monitoring

```
ps -p $(cat logs/pid/qwen2.5-32b/${INFER_RUN_ID}.pid)
```
```
tail -f logs/infer/qwen2.5-32b/${INFER_RUN_ID}.log
```

Important:
- Keep train/test split fixed across GPT and OSS comparisons.
- Use only train-set for training (test-set is for evaluation only).
- Do not commit large artifacts, logs, checkpoints, or secrets. Keep `.gitignore` updated.

## Code Organization

Model-specific scripts are organized under:
- `code/fine-tuning/{gpt,qwen2.5-32b,qwen2.5-7b,gptoss,gemma-3-27b,gemma-3-4b,llama-3.1-8b,llama-3.1-70b}`
- `code/single-node/{gpt,qwen2.5-32b,qwen2.5-7b,gptoss,gemma-3-27b,gemma-3-4b,llama-3.1-8b,llama-3.1-70b}`
- `code/multi-node/{gpt,qwen,gemma}`

Common single-node evaluation script: `code/single-node/evaluation.py`.
Most model-specific production scripts are under each `gpt` directory. Versioned OSS model directories sit under the `code/fine-tuning` and `code/single-node` trees.