"""Local Gradio interface. Run with `uv run --locked python app.py`."""

import os

# Set before importing Gradio, including when this module is imported by tests.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_RUN_HISTORY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

import logging
from pathlib import Path
import secrets

import gradio as gr
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse

from backend.config import MAX_QUESTION_CHARS, Settings, get_settings
from backend.council import CouncilError, iterate_council
from backend.openrouter import ModelCheckError, validate_model_configuration

logger = logging.getLogger(__name__)


class LocalRequestMiddleware:
    """Reject foreign browser origins, file routes, and oversized request bodies."""

    def __init__(self, app, port: int):
        self.app = app
        self.origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        origin = headers.get(b"origin")
        if origin is not None and origin.decode("latin-1") not in self.origins:
            return await PlainTextResponse("Origin not allowed", status_code=403)(scope, receive, send)
        path = scope["path"]
        if path.startswith(("/gradio_api/upload", "/gradio_api/file=", "/gradio_api/proxy=")):
            return await PlainTextResponse("Not found", status_code=404)(scope, receive, send)
        if scope["method"] in {"POST", "PUT", "PATCH"}:
            # Count actual bytes too: Content-Length can be missing or misleading.
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > 65_536:
                    return await PlainTextResponse("Request too large", status_code=413)(scope, receive, send)
                if not message.get("more_body", False):
                    break

            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            return await self.app(scope, bounded_receive, send)
        return await self.app(scope, receive, send)


async def answer_question(question: str):
    settings = get_settings()
    answers, reviews = {}, {}
    final, mapping, aggregate = "", "", []
    warnings = []
    diagnostics = []

    def snapshot(status):
        if warnings:
            status += "\n" + "\n".join(warnings)
        return (
            status, final, mapping, aggregate,
            *(answers.get(m, "") for m in settings.models),
            *(reviews.get(m, "") for m in settings.models),
            format_diagnostics(diagnostics),
        )

    try:
        async for event in iterate_council(question):
            if "diagnostics" in event:
                diagnostics = event["diagnostics"]
            kind = event["type"]
            if kind == "preflight_start":
                status = "Checking model IDs and low-reasoning support (no generation)…"
            elif kind == "diagnostics":
                continue
            elif kind == "stage1_start":
                status = "1/3 — Collecting independent answers…"
            elif kind == "stage1_complete":
                answers = {r["model"]: r["response"] for r in event["data"]}
                for model in settings.models:
                    if model not in answers:
                        detail = next((d["error"] for d in diagnostics if d["model"] == model and d["stage"] == "Answers"), None)
                        answers[model] = detail or "This model did not return an answer."
                        warnings.append(f"No answer from {model}.")
                for item in diagnostics:
                    warnings.extend(f"{item['model']}: {warning}" for warning in item["warnings"])
                status = "Independent answers received."
            elif kind == "stage2_start":
                status = "2/3 — Reviewing anonymous answers…"
            elif kind == "stage2_complete":
                labels = event["metadata"]["label_to_model"]
                mapping = "\n".join(f"{label} = {model}" for label, model in labels.items())
                for r in event["data"]:
                    ballot = r["parsed_ranking"]
                    extracted = " > ".join(labels[label] for label in ballot) if ballot else "Invalid or incomplete; excluded from averages."
                    reviews[r["model"]] = r["ranking"] + "\n\nExtracted ranking: " + extracted
                for model in settings.models:
                    if model not in reviews:
                        detail = next((d["error"] for d in diagnostics if d["model"] == model and d["stage"] == "Reviews"), None)
                        reviews[model] = detail or "Review skipped or unavailable; see diagnostics."
                        warnings.append(f"No review from {model}.")
                aggregate = [[r["model"], r["average_rank"], r["rankings_count"]]
                             for r in event["metadata"]["aggregate_rankings"]]
                if not aggregate:
                    warnings.append("No valid complete rankings were returned; averages are unavailable.")
                status = "Peer reviews received."
            elif kind == "stage3_start":
                status = "3/3 — Preparing the chairman's answer…"
            elif kind == "stage3_complete":
                final = event["data"]["response"]
                status = "Complete."
            else:
                continue
            yield snapshot(status)
    except CouncilError as exc:
        if not final:
            final = "No final synthesis was produced.\n\n" + str(exc)
        yield snapshot(str(exc))
    except Exception:
        # Do not leak exception messages, headers, prompts or provider responses.
        logger.error("Council stopped because of an unexpected internal error.")
        yield snapshot("The run stopped because of an internal error. No retry was made.")


def format_diagnostics(diagnostics: list[dict]) -> str:
    rows = []
    for item in diagnostics:
        fields = [f"{item['stage']} | {item['model']}"]
        if item.get("finish_reason"):
            fields.append(f"finish_reason={item['finish_reason']}")
        for name, value in item.get("usage", {}).items():
            fields.append(f"{name}={value}")
        rows.append(" | ".join(fields))
        rows.append(item.get("error") or "Answer received.")
        rows.extend(item.get("warnings", []))
        rows.append("")
    return "\n".join(rows)


async def check_models():
    try:
        catalog = await validate_model_configuration()
        return (f"Model IDs and low-reasoning support checked for {len(catalog)} models. "
                "No generation requests were made. Provider availability under ZDR rules is checked at generation time.")
    except ModelCheckError as exc:
        return str(exc)


def create_app() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(title="LLM Council — Local", analytics_enabled=False) as demo:
        gr.Markdown("# LLM Council\nIndependent answers → anonymous peer review → final synthesis.")
        gr.Markdown(
            f"Questions are sent to your configured OpenRouter models. "
            f"Each run makes up to **{2 * len(settings.models) + 1} paid requests**, "
            f"capped at **{settings.max_output_tokens:,} output tokens per request**. "
            "Reasoning effort is low for models that support it. "
            "Results are kept in this page only."
        )
        question = gr.Textbox(
            label="Your question", lines=5, max_lines=12, max_length=MAX_QUESTION_CHARS,
            placeholder="Describe the problem you want the council to solve…",
            html_attributes={"autocomplete": "off"},
        )
        run = gr.Button("Consult the council", variant="primary")
        check = gr.Button("Check model IDs (no generation)")
        status = gr.Textbox(label="Progress", value="Ready.", interactive=False, lines=2, max_lines=8)
        with gr.Tabs():
            with gr.Tab("Final answer"):
                final = gr.Textbox(label=f"Chairman: {settings.chairman}", interactive=False,
                                   lines=18, max_lines=40, buttons=["copy"])
            with gr.Tab("Individual answers"):
                answer_boxes = []
                with gr.Tabs():
                    for model in settings.models:
                        with gr.Tab(model):
                            answer_boxes.append(gr.Textbox(label=model, interactive=False,
                                                           lines=18, max_lines=40, buttons=["copy"]))
            with gr.Tab("Peer review"):
                mapping = gr.Textbox(label="Anonymous labels used by the reviewers", interactive=False, lines=4)
                aggregate = gr.Dataframe(headers=["Model", "Average position (lower is better)", "Valid votes"],
                                         datatype=["str", "number", "number"], interactive=False)
                review_boxes = []
                with gr.Tabs():
                    for model in settings.models:
                        with gr.Tab(model):
                            review_boxes.append(gr.Textbox(label=f"Review by {model}", interactive=False,
                                                           lines=18, max_lines=40, buttons=["copy"]))
            with gr.Tab("Diagnostics"):
                diagnostic_box = gr.Textbox(label="Per-model status and reported usage (cost in USD)",
                                            interactive=False, lines=16, max_lines=40)
        outputs = [status, final, mapping, aggregate, *answer_boxes, *review_boxes, diagnostic_box]
        run.click(answer_question, inputs=[question], outputs=outputs, api_visibility="private",
                  concurrency_limit=1, concurrency_id="council", trigger_mode="once")
        check.click(check_models, outputs=[diagnostic_box], api_visibility="private",
                    concurrency_limit=1, concurrency_id="council", trigger_mode="once")
        gr.ClearButton([question, *outputs], value="Clear this page")
        gr.Markdown("Answers are displayed as plain text. Check important claims independently.")
    return demo.queue(default_concurrency_limit=1, max_size=1, api_open=False)


def launch_options(settings: Settings, password: str) -> dict:
    return {
        "server_name": "127.0.0.1",
        "server_port": settings.port,
        "share": False,
        "inbrowser": False,
        "auth": ("council", password),
        "auth_message": "Use the local login credentials printed in your terminal.",
        "strict_cors": True,
        "show_error": False,
        "footer_links": [],
        "run_history": False,
        "ssr_mode": False,
        "mcp_server": False,
        "pwa": False,
        "enable_monitoring": False,
        "allowed_paths": [],
        "blocked_paths": [str(Path(Path.cwd().anchor))],
        "max_file_size": 1024,
        "theme": gr.themes.Default(font=["system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"]),
        "app_kwargs": {
            "docs_url": None, "redoc_url": None, "openapi_url": None,
            "middleware": [
                Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"]),
                Middleware(LocalRequestMiddleware, port=settings.port),
            ],
        },
    }


def main():
    try:
        settings = get_settings()
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None
    if not settings.api_key or settings.api_key == "your-openrouter-key-here":
        raise SystemExit("Set OPENROUTER_API_KEY in the project's .env file before starting.")
    password = settings.local_password or secrets.token_urlsafe(24)
    print(f"\nOpen http://127.0.0.1:{settings.port}\nUsername: council", flush=True)
    if settings.local_password:
        print("Password: the LOCAL_APP_PASSWORD configured in your environment\n", flush=True)
    else:
        print(f"Temporary password: {password}\n", flush=True)
    create_app().launch(**launch_options(settings, password))


if __name__ == "__main__":
    main()
