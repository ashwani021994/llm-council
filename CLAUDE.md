# Development notes

This is a Python-only, local Gradio app. Read README.md for setup and security scope.
The previous React/Vite frontend, public FastAPI routes, and JSON conversation storage
have been removed. Do not restore those entry points incidentally.

- `app.py`: Gradio UI, plain-text model outputs, generated login password, loopback
  launch settings, origin/Host validation, request-size limits, file/upload denial.
- `backend/config.py`: settings from this project's `.env` or process environment.
  Keys and passwords must never appear in browser configuration or object reprs.
- `backend/openrouter.py`: fixed HTTPS destination, bounded output and response size,
  no redirects, no env proxies, no retries, no tools, ZDR/data-policy enforcement.
  Public model-catalog preflight rejects missing models and models whose advertised
  effort list excludes low. Request low reasoning, including for mandatory-reasoning
  models; omit unsupported parameters for non-reasoning models. Never use raw
  reasoning as a final answer. Return safe status and numeric usage diagnostics for
  empty or truncated responses, rather than discarding billed completion metadata.
- `backend/council.py`: async stage functions, strict ballot validation, full-run and
  incremental interfaces. A process-wide lock limits complete runs to one at a time.
- `tests/test_council.py`: unittest tests with mocked network calls and an ASGI client.
  Run `uv run --locked python -m unittest discover -s tests -v`.

Preserve local-only defaults. Deployment is out of scope for now. Do not call paid
providers during development verification unless the user asks for a live test.
Never log secrets, prompts, provider bodies, or raw upstream exceptions.
No saved conversations or browser run history. Model outputs must remain plain text.
Keep Python dependencies locked and audit intentional updates.
