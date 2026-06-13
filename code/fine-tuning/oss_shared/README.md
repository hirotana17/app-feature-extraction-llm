# OSS Fine-tuning Shared Components

This directory contains shared scripts for OSS LLM fine-tuning:

- `prepare_sft_dataset.py`: Converts `formatted_original_data/*.json` to SFT JSONL.
- `train_unsloth_lora.py`: Runs Unsloth + PEFT LoRA training.
- `config.py`: Model presets for versioned Qwen, GPT-OSS, Gemma, and Llama variants.

## Usage

Prepare SFT data (default: split train into 90/10 for train/validation):

```bash
python code/fine-tuning/oss_shared/prepare_sft_dataset.py -d ./data/in-domain/bin0/
```

Output paths:
- Train JSONL: `fine_tuning_data/train-set-sft-promptft1.jsonl`
- Validation JSONL: `fine_tuning_data/validation-set-sft-promptft1.jsonl`

Run training via wrappers:

```bash
python code/fine-tuning/qwen2.5-32b/train.py -d ./data/in-domain/bin0/  # Qwen2.5-32B
python code/fine-tuning/qwen2.5-7b/train.py -d ./data/in-domain/bin0/   # Qwen2.5-7B
python code/fine-tuning/gptoss/train.py -d ./data/in-domain/bin0/
python code/fine-tuning/gemma-3-27b/train.py -d ./data/in-domain/bin0/  # Gemma-3-27B
python code/fine-tuning/gemma-3-4b/train.py -d ./data/in-domain/bin0/   # Gemma-3-4B
python code/fine-tuning/llama-3.1-8b/train.py -d ./data/in-domain/bin0/  # Llama-3.1-8B
python code/fine-tuning/llama-3.1-70b/train.py -d ./data/in-domain/bin0/ # Llama-3.1-70B
```

Inference and evaluation use `formatted_original_data/test-set.json` directly.
