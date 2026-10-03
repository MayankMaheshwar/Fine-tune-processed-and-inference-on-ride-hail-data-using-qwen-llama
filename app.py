"""Local chat and evaluation UI for the base and fine-tuned MLX models."""

from __future__ import annotations

import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

import gradio as gr


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/qwen3.5-2b-mlx-4bit"
ADAPTER_V1_PATH = ROOT / "adapters/ride_hailing_v1"
ADAPTER_V2_PATH = ROOT / "adapters/ride_hailing_v2"
EVAL_PATH = ROOT / "data/eval/ride_hailing_cases.json"
LOGGER = logging.getLogger(__name__)

MIN_MAX_TOKENS = 64
MAX_MAX_TOKENS = 384
MAX_USER_MESSAGE_CHARS = 4_000
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 8_000
MAX_CONTEXT_CHARS = 12_000

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

BASE_MODEL_LABEL = "Qwen3.5 2B · base"
ADAPTER_CANDIDATES = (
    ("Qwen3.5 2B · adapter v1 (200 steps)", ADAPTER_V1_PATH),
    ("Qwen3.5 2B · adapter v2 (100 steps)", ADAPTER_V2_PATH),
)


def available_model_options(
    adapter_candidates: tuple[tuple[str, Path], ...] = ADAPTER_CANDIDATES,
) -> dict[str, Path | None]:
    options: dict[str, Path | None] = {BASE_MODEL_LABEL: None}
    for label, path in adapter_candidates:
        if (path / "adapters.safetensors").is_file():
            options[label] = path
    return options


MODEL_OPTIONS = available_model_options()
DEFAULT_MODEL = next(reversed(MODEL_OPTIONS))

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

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            "Base model is missing. Download it using the Install instructions in README.md."
        )
    if adapter_path is not None and not (adapter_path / "adapters.safetensors").is_file():
        raise FileNotFoundError(
            "This adapter is no longer available. Restart the app after installing it."
        )

    if _active_model is not None:
        old_model = _active_model
        old_tokenizer = _active_tokenizer
        _active_model = None
        _active_tokenizer = None
        _active_model_key = None
        del old_model
        del old_tokenizer
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


def _validate_max_tokens(value: Any) -> int:
    try:
        max_tokens = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Choose a response length between 64 and 384 tokens.") from exc
    if not MIN_MAX_TOKENS <= max_tokens <= MAX_MAX_TOKENS:
        raise ValueError("Choose a response length between 64 and 384 tokens.")
    return max_tokens


def _clean_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        return "The model returned unfinished internal reasoning, so this output was suppressed. Try again."
    text = text.replace("<|channel|>final", "").strip()
    return text or "The model returned an empty response. Try again."


def generate_answer(choice: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    max_tokens = _validate_max_tokens(max_tokens)
    if not isinstance(messages, list):
        raise ValueError("Conversation history is invalid. Clear the conversation and try again.")

    model, tokenizer = _load_model(choice)
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            content = content.strip()
            if len(content) > MAX_USER_MESSAGE_CHARS:
                raise ValueError("Each message must be 4,000 characters or fewer.")
            context_chars += len(content)
            if context_chars > MAX_CONTEXT_CHARS:
                raise ValueError("Conversation is too long. Clear it and start a new chat.")
            prompt_messages.append({"role": role, "content": content})
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
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.2),
        verbose=False,
    )
    return _clean_generated_text(result)


def _history_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if not isinstance(history, list):
        return messages
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                content = content.strip()
                if len(content) <= MAX_USER_MESSAGE_CHARS:
                    messages.append({"role": item["role"], "content": content})
    while sum(len(item["content"]) for item in messages) > MAX_HISTORY_CHARS:
        messages.pop(0)
    return messages


def chat(message: str, history: list[dict[str, Any]], choice: str, max_tokens: int):
    if not isinstance(message, str):
        raise gr.Error("Enter a text message.")
    message = message.strip()
    if not message:
        return _history_messages(history), ""
    if len(message) > MAX_USER_MESSAGE_CHARS:
        raise gr.Error("Keep each message to 4,000 characters or fewer.")
    messages = _history_messages(history) + [{"role": "user", "content": message}]
    try:
        answer = generate_answer(choice, messages, max_tokens)
    except (FileNotFoundError, ValueError) as exc:
        raise gr.Error(str(exc)) from None
    except Exception:
        LOGGER.exception("Local chat generation failed")
        raise gr.Error("The local model failed. Check the app terminal for details.") from None
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
    if not isinstance(case_id, str) or case_id not in CASE_BY_ID:
        raise gr.Error("Choose one of the available evaluation scenarios.")
    case = CASE_BY_ID[case_id]
    try:
        answer = generate_answer(
            choice,
            [{"role": "user", "content": case["prompt"]}],
            max_tokens,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise gr.Error(str(exc)) from None
    except Exception:
        LOGGER.exception("Evaluation scenario generation failed")
        raise gr.Error("The local model failed. Check the app terminal for details.") from None
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
        except (FileNotFoundError, ValueError) as exc:
            raise gr.Error(str(exc)) from None
        except Exception:
            LOGGER.exception("Evaluation batch failed on case %s", case["id"])
            raise gr.Error("The local model failed. Check the app terminal for details.") from None
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
            value=DEFAULT_MODEL,
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
        send.click(
            chat,
            [message, chatbot, model_choice, max_tokens],
            [chatbot, message],
            concurrency_limit=1,
            concurrency_id="local-model",
        )
        message.submit(
            chat,
            [message, chatbot, model_choice, max_tokens],
            [chatbot, message],
            concurrency_limit=1,
            concurrency_id="local-model",
        )
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
            concurrency_limit=1,
            concurrency_id="local-model",
        )
        run_all = gr.Button("Run all 12 scenarios")
        all_results = gr.Dataframe(
            headers=["Case", "Scenario", "Review criteria", "Model response"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            label="Review each response against its criteria",
        )
        run_all.click(
            run_all_cases,
            [model_choice, max_tokens],
            [all_results],
            concurrency_limit=1,
            concurrency_id="local-model",
        )

    demo.load(get_case_details, [case_id], [case_prompt, case_checks])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=16)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_error=False,
    )
