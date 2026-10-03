# Local ride-hailing support model

A local-first project to compare a small language model, fine-tune it for customer-support responses, and inspect the results in a local UI.

## Current setup

- Machine: Apple M2 Pro, 16 GB unified memory
- Python: 3.13 (`.venv`)
- Training runtime: MLX-LM 0.31.3
- Comparison model: the already-installed `llama3:latest` in Ollama (8B, Q4_0)
- Fine-tuning candidate: `mlx-community/Qwen3.5-2B-MLX-4bit` (Apache-2.0; approximately 1.75 GB)

## Setup

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Planned workflow

1. Download and load the MLX checkpoint. **Done.**
2. Inspect the public customer-support dataset and its license. **Done.** The Bitext file is licensed under CDLA-Sharing-1.0. Its training results are not subject to that agreement's data-sharing condition; if we publish adapted data, the agreement requires sharing it under the same terms. See <https://cdla.dev/sharing-1-0/>.
3. Prepare general support examples and add clearly labeled synthetic ride-hailing examples. **Done.**
4. Fine-tune LoRA adapters and compare them with the untuned checkpoint. **Done: v1 saved at step 200; v2 saved at step 100.**
5. Build a local UI for conversations and repeatable evaluation cases.

## Data preparation

Download the public CSV and prepare the local training splits:

```sh
.venv/bin/hf download bitext/Bitext-customer-support-llm-chatbot-training-dataset Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv --repo-type dataset --local-dir data/raw/bitext
.venv/bin/python scripts/prepare_data.py --output data/processed/ride_hailing_v1
```

The 12 manual evaluation cases in `data/eval/ride_hailing_cases.json` are held out from training. The seed conversations in `data/synthetic/ride_hailing.jsonl` are hypothetical examples, not Ola or Uber transcripts or official policies.

## Fine-tuning

```sh
.venv/bin/mlx_lm.lora --config configs/train_ride_hailing.yaml
```

The current config uses the more ride-hailing-focused v2 split and runs for 100 steps, matching the saved v2 checkpoint. The v1 adapter saved at step 200; macOS stopped that training process at step 220. The first v2 attempt saved at step 100 and was stopped by the same Metal interactivity error at step 160. Both saved adapters are available for comparison. The UI labels the saved step count for each.

## Local UI

```sh
.venv/bin/python app.py
```

Open <http://127.0.0.1:7860>. The UI switches between the base and adapter, supports a local chat, and runs held-out scenarios with manual review criteria. It does not call Ola, Uber, or any external model service.

Downloaded model weights, datasets, and adapters stay out of version control. Public data will not be treated as real Ola or Uber policy.
