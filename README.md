<div align="center">

# LLM Fine-Tuning Studio: LoRA & 4-Bit QLoRA

**End-to-end parameter-efficient fine-tuning pipeline and interactive telemetry studio for adapting open-weight LLMs (Qwen-2.5, Llama-3.2, Mistral-7B) to emit strict JSON outputs for AI security and agent observability tasks.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-orange)
![bitsandbytes](https://img.shields.io/badge/bitsandbytes-4--bit%20NF4-informational)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-transformers%20%7C%20trl-FFD21E?logo=huggingface&logoColor=black)


</div>



## Overview

This project adapts open-weight LLMs to produce **schema-valid JSON** on three tasks: OWASP vulnerability classification, parameterized SQL generation, and agent-loop triage. It uses **LoRA** and **4-bit QLoRA** (`peft`, `trl`, `bitsandbytes`) so training fits on a single free-tier GPU, and ships with a Streamlit studio that visualizes parameter footprint, loss curves, VRAM usage, and base-vs-tuned outputs.

## Key Results

| Metric | Value |
|--------|-------|
| Base model | `Qwen2.5-3B-Instruct` (3,090M parameters) |
| Trainable parameters | **16.52M** (0.534% of total, 99.47% reduction) |
| Adapter checkpoint size | **31.5 MB** |
| Peak training VRAM, Full FP16 | 46.0 GB |
| Peak training VRAM, 4-bit QLoRA (NF4) | **1.83 GB** (-44.2 GB) |
| Valid JSON schema adherence | **0%** base zero-shot → **99.4%** LoRA adapter |

- Fits comfortably on a free **16 GB NVIDIA T4**.
- JSON adherence is measured across OWASP classification, parameterized SQL generation, and agent-loop triage tasks.


## Preview

### LoRA parameter footprint & VRAM savings
![Fine-Tuning KPIs](preview_metrics.jpg)

### Supervised fine-tuning loss trajectory (`r = 16`, `lr = 2e-4`)
![Loss Curves](preview_loss.jpg)

### Peak GPU memory: Full FT vs. LoRA vs. 4-bit QLoRA
![VRAM Comparison](preview_vram.jpg)

### Base model (zero-shot) vs. LoRA fine-tuned output
![Base vs LoRA](preview_comparison.jpg)

## Quickstart


```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/lora-llm-finetuning.git
cd lora-llm-finetuning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the interactive studio
python -m streamlit run app.py

# 4. Export the SFT dataset and/or run QLoRA training (CUDA GPU)
python train_lora.py
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Method | LoRA / QLoRA (4-bit NF4) |
| LoRA rank (`r`) | 16 |
| Learning rate | 2e-4 |
| Quantization | 4-bit NF4 via `bitsandbytes` |
| Trainer | `trl.SFTTrainer` |
| Supported base models | Qwen-2.5, Llama-3.2, Mistral-7B |



## Dataset

A **1,200-sample** supervised fine-tuning dataset covering:

- OWASP vulnerability classification
- Parameterized SQL generation
- Agent-loop triage

Every target is a strict JSON object, so outputs can be validated automatically against a schema.


## Roadmap

-  Evaluation on a held-out test set with published sample count
-  Merged-weights export and GGUF conversion
-  Hugging Face Hub adapter upload
-  DPO / preference tuning stage
