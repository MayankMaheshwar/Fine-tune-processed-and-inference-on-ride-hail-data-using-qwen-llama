# Local ride-hailing support model lab

A local prototype for preparing customer-support data, fine-tuning a small model with MLX LoRA, and reviewing its answers in a browser UI. It is not affiliated with Ola, Uber, or another ride-hailing company, and it is not ready for customer-facing use.

## Project status

- Tested on an Apple M2 Pro Mac with 16 GB unified memory.
- Base model: [Qwen3.5-2B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-4bit), about 1.7 GB on disk, Apache-2.0.
- Fine-tuning: MLX-LM 0.31.3 with LoRA. The current configuration trains the v2 adapter for 100 iterations.
- Data: a public, general customer-support dataset plus explicitly synthetic ride-hailing examples. There are no Ola or Uber call transcripts or official company policies in this project.
- UI: local chat and 12 held-out scenarios with manual review criteria.

The v2 adapter improved some safety and privacy responses in the included scenarios, but the model can still make mistakes. Review every answer; do not use it to handle real customer cases.

The ride-hailing-specific training set contains 31 hypothetical examples, repeated 24 times for this experiment. This small dataset can cause overfitting and is not evidence of reliable real-world performance.

## Requirements

- Apple Silicon Mac with a compatible macOS/Metal setup for MLX
- Python 3.13
- Internet access to download the model and public dataset
- Free disk space for the Python environment, model, dataset, and generated adapters

Ollama is not required. Although Ollama's `llama3:latest` was installed during development, the current UI and evaluation script compare the Qwen base model and Qwen adapters only.

## Install

From the repository root:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Download the base model. Model files are local and are not committed to Git:

```sh
.venv/bin/hf download mlx-community/Qwen3.5-2B-MLX-4bit \
  --local-dir models/qwen3.5-2b-mlx-4bit
```

## Prepare data

Download the Bitext dataset. The source CSV is also kept out of Git:

```sh
.venv/bin/hf download bitext/Bitext-customer-support-llm-chatbot-training-dataset \
  Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv \
  --repo-type dataset \
  --local-dir data/raw/bitext
```

Prepare the v2 split used by the current training configuration:

```sh
.venv/bin/python scripts/prepare_data.py \
  --output data/processed/ride_hailing_v2
```

The script splits the general Bitext examples into train, validation, and test sets, then mixes general training rows with repeated synthetic ride-hailing examples. The 12 scenarios in `data/eval/ride_hailing_cases.json` are separate manual evaluation cases. They are not company policies.

## Fine-tune

```sh
.venv/bin/mlx_lm.lora --config configs/train_ride_hailing.yaml
```

The config reads `data/processed/ride_hailing_v2` and writes the adapter to `adapters/ride_hailing_v2`. The earlier v1 adapter was saved at step 200 from an interrupted experiment; it is not recreated by the current v2 config. Model and adapter files are excluded from Git. On a fresh checkout, the UI lists the base model; the v2 option appears after training and restarting the app.

## Run the evaluation

Run the base model and the v2 adapter against the 12 manual scenarios:

```sh
.venv/bin/python scripts/evaluate_cases.py \
  --models "Qwen3.5 2B · base" "Qwen3.5 2B · adapter v2 (100 steps)"
```

The script writes a local JSON report under `data/eval/generated/`. Generated reports are excluded from Git. The review criteria support human inspection; they are not an automated safety score.

## Launch the UI

```sh
.venv/bin/python app.py
```

Open <http://127.0.0.1:7860>. The app uses the local MLX model and does not call Ola, Uber, or an external model-inference service. It lists only adapters found on disk when the app starts.

## Local use and security

The server binds to `127.0.0.1` and Gradio sharing is disabled. The app has no authentication and is intended for single-user local use. Do not expose its port to a network, change the bind address to `0.0.0.0`, or enable public sharing. Chat messages are sent from the browser to the local process for inference; this prototype does not write chat history to project files.

## Data and model licenses

The [Bitext dataset card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) identifies the data as CDLA-Sharing-1.0. Review the [license terms](https://cdla.dev/sharing-1-0/) before redistributing the source data or modified versions. Section 3.5 describes conditions for “Results,” with the license defining Results to exclude more than a de minimis portion of the source data. This README does not determine whether any particular adapter or generated artifact meets that definition.

The Qwen checkpoint has its own Apache-2.0 terms, described on its [model card](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-4bit). Those terms do not set a license for this project's source code. This repository currently has no top-level project license.

Downloaded model weights, adapters, the Bitext CSV, processed Bitext-derived splits, and generated evaluation reports are ignored by Git. The synthetic ride-hailing examples and manual evaluation cases are included in the repository; they are hypothetical and are not real Ola or Uber transcripts or policy statements.
