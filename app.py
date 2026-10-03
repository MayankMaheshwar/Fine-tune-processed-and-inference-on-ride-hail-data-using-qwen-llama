"""Local chat and evaluation UI for the base and fine-tuned MLX models."""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

import gradio as gr


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/qwen3.5-2b-mlx-4bit"
ADAPTER_V1_PATH = ROOT / "adapters/ride_hailing_v1"
ADAPTER_V2_PATH = ROOT / "adapters/ride_hailing_v2"
EVAL_PATH = ROOT / "data/eval/ride_hailing_cases.json"

SYSTEM_PROMPT = (
    "You are a general assistant for hypothetical ride-hailing support. "
    "You are not affiliated with a named company and cannot access accounts or take actions. "
    "Never invent or infer fees, refund eligibility, refund timing, phone numbers, or company policy. "
    "For charges or refunds, direct the customer to the trip receipt and the official app's trip Help option; "
    "say that the service must review the trip before confirming an outcome. Never say a charge should "
    "be refundable. Do not claim to issue refunds, change trips, or file reports. Do not request passwords, "
    "one-time codes, or full payment details. For immediate danger, put local emergency services and "
    "moving to safety first; app safety tools are an additional option. Keep the response concise and do not "
    "show internal reasoning."
)

MODEL_OPTIONS = {
    "Qwen3.5 2B · base": None,
    "Qwen3.5 2B · adapter v1 (200 steps)": ADAPTER_V1_PATH,
    "Qwen3.5 2B · adapter v2 (100 steps)": ADAPTER_V2_PATH,
}

_active_model_key: str | None = None
_active_model: Any = None
_active_tokenizer: Any = None


def _load_model(choice: str) -> tuple[Any, Any]:
    global _active_model_key, _active_model, _active_tokenizer

    if choice not in MODEL_OPTIONS:
        raise ValueError("Choose one of the available Qwen model versions.")
    adapter_path = MODEL_OPTIONS[choice]
    key = f"{choice}:{adapter_path or 'base'}"
    if key == _active_model_key:
        return _active_model, _active_tokenizer

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if adapter_path is not None and not (adapter_path / "adapters.safetensors").exists():
        raise FileNotFoundError(f"Fine-tuned adapter not found: {adapter_path}")

    if _active_model is not None:
        del _active_model
        del _active_tokenizer
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()

    from mlx_lm import load

    if adapter_path is None:
        model, tokenizer = load(str(MODEL_PATH))
    else:
        model, tokenizer = load(str(MODEL_PATH), adapter_path=str(adapter_path))
    _active_model_key = key
    _active_model = model
    _active_tokenizer = tokenizer
    return model, tokenizer


def _clean_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        return "The model returned unfinished internal reasoning, so this output was suppressed. Try again."
    text = text.replace("<|channel|>final", "").strip()
    return text or "The model returned an empty response. Try again."


def generate_answer(choice: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    model, tokenizer = _load_model(choice)
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            prompt_messages.append({"role": role, "content": content.strip()})
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=int(max_tokens),
        sampler=make_sampler(temp=0.2),
        verbose=False,
    )
    return _clean_generated_text(result)


def _history_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in (history or [])[-12:]:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            content = item.get("content")
            if isinstance(content, str):
                messages.append({"role": item["role"], "content": content})
    return messages


def chat(message: str, history: list[dict[str, Any]], choice: str, max_tokens: int):
    if not message.strip():
        return history, ""
    messages = _history_messages(history) + [{"role": "user", "content": message.strip()}]
    try:
        answer = generate_answer(choice, messages, max_tokens)
    except Exception as exc:  # Show actionable local setup errors in the UI.
        raise gr.Error(f"Local model error: {exc}") from exc
    return messages + [{"role": "assistant", "content": answer}], ""


def load_cases() -> list[dict[str, Any]]:
    with EVAL_PATH.open("r", encoding="utf-8") as source:
        return json.load(source)


CASES = load_cases()
CASE_BY_ID = {case["id"]: case for case in CASES}


def get_case_details(case_id: str) -> tuple[str, str]:
    case = CASE_BY_ID[case_id]
    checks = "\n".join(f"- [ ] {item}" for item in case["checks"])
    return case["prompt"], checks


def run_case(case_id: str, choice: str, max_tokens: int) -> tuple[str, str]:
    case = CASE_BY_ID[case_id]
    try:
        answer = generate_answer(
            choice,
            [{"role": "user", "content": case["prompt"]}],
            max_tokens,
        )
    except Exception as exc:
        raise gr.Error(f"Local model error: {exc}") from exc
    return answer, "\n".join(f"- [ ] {item}" for item in case["checks"])


def run_all_cases(choice: str, max_tokens: int):
    rows = []
    for case in CASES:
        try:
            answer = generate_answer(
                choice,
                [{"role": "user", "content": case["prompt"]}],
                max_tokens,
            )
        except Exception as exc:
            raise gr.Error(f"Local model error on {case['id']}: {exc}") from exc
        rows.append(
            [case["id"], case["prompt"], "\n".join(case["checks"]), answer]
        )
    return rows


with gr.Blocks(title="Ride-hailing Support Model Lab") as demo:
    gr.Markdown(
        "# Ride-hailing Support Model Lab\n"
        "Runs locally on this Mac. These examples are synthetic and do not define "
        "Ola, Uber, or any other company's policies. Review every response before use."
    )
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=list(MODEL_OPTIONS),
            value=list(MODEL_OPTIONS)[2],
            label="Model version",
        )
        max_tokens = gr.Slider(64, 384, value=192, step=32, label="Maximum response tokens")

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=440, label="Conversation")
        with gr.Row():
            message = gr.Textbox(
                placeholder="Try a ride-hailing support scenario…",
                label="Customer message",
                scale=8,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear conversation")
        send.click(chat, [message, chatbot, model_choice, max_tokens], [chatbot, message])
        message.submit(chat, [message, chatbot, model_choice, max_tokens], [chatbot, message])
        clear.click(lambda: ([], ""), outputs=[chatbot, message])

    with gr.Tab("Evaluation cases"):
        case_id = gr.Dropdown(
            choices=[case["id"] for case in CASES],
            value=CASES[0]["id"],
            label="Held-out scenario",
        )
        case_prompt = gr.Textbox(label="Scenario", interactive=False, lines=2)
        case_checks = gr.Markdown(label="Review criteria")
        case_id.change(get_case_details, [case_id], [case_prompt, case_checks])
        run_selected = gr.Button("Run selected scenario", variant="primary")
        case_answer = gr.Textbox(label="Model response", lines=6, interactive=False)
        selected_checks = gr.Markdown(label="Manual review checklist")
        run_selected.click(
            run_case,
            [case_id, model_choice, max_tokens],
            [case_answer, selected_checks],
        )
        run_all = gr.Button("Run all 12 scenarios")
        all_results = gr.Dataframe(
            headers=["Case", "Scenario", "Review criteria", "Model response"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            label="Review each response against its criteria",
        )
        run_all.click(run_all_cases, [model_choice, max_tokens], [all_results])

    demo.load(get_case_details, [case_id], [case_prompt, case_checks])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
