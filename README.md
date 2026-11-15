# LLM Council — local Gradio app

A Python-only app that asks several LLMs to solve a problem, has them review
anonymous candidate answers, and asks a chairman model to synthesize a final answer.
The UI shows each stage, the original reviews, extracted rankings, and average ranks.
No Node, npm, separate frontend build, or separate API server is required.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. The project
uses Python 3.12; uv can install it if necessary.

```bash
uv sync --locked
cp .env.example .env
chmod 600 .env
```

Edit `.env` and replace `your-openrouter-key-here` with a **dedicated OpenRouter key
with a small credit limit**. Keep the key out of the question box and source control.
Then start the app:

```bash
./start.sh
# Equivalent: uv run --locked python app.py
```

Open **http://127.0.0.1:7860**. Log in as **council** using the temporary password
printed in the terminal. A different random password is generated each time the app
starts. To keep a fixed password, set `LOCAL_APP_PASSWORD` in `.env` (at least 16
characters; use a password different from the API key).

Enter a question and click **Consult the council**. Stage updates appear as the
models finish. Inspect **Final answer**, **Individual answers**, and **Peer review**.
Use **Check model IDs (no generation)** to check model IDs and low-reasoning support
without a paid inference request. This check also runs automatically before each run.
The **Diagnostics** tab shows per-model errors, finish reasons, token usage, and the
cost OpenRouter reports. Missing usage/cost fields are unavailable, not assumed zero.
Use **Clear this page** to clear visible results. Press Ctrl+C in the terminal to stop.

## Models and limits

Optional settings in `.env`:

```dotenv
COUNCIL_MODELS=openai/gpt-4.1-mini,openai/gpt-4.1-nano
CHAIRMAN_MODEL=openai/gpt-4.1-mini
MAX_OUTPUT_TOKENS=1024
LOCAL_PORT=7860
```

Restart after changing configuration. Configure 2–6 distinct council models. The
defaults use two non-reasoning models; availability is checked against OpenRouter's
catalog at runtime. Existing explicit `.env` model selections are not overwritten.
Model selection is server-side, not controlled by the browser.

- The default two-member council makes **up to five model calls per question**:
  two independent answers, two reviews, and one chairman call. Four members make
  up to nine calls. There is no extra
  title-generation call, retry, agent tool, code execution, or web browsing step.
- Questions are limited to 8,000 characters. Each call has a 1,024 output-token cap
  by default, configurable from 128 to 4,096, a 90-second timeout, and response-size
  limits. A complete run has a five-minute timeout.
- Only one run executes at a time. The UI queue holds at most one waiting request.
- Failed initial models are not called again for review. If fewer than two initial
  answers succeed, the run stops before review and synthesis to avoid further charges.
  Completed answers stay visible. If the chairman fails, completed answers
  and reviews remain visible.
- Duplicate, unknown, or incomplete ranking ballots are excluded from averages.
  These rankings are opinions, not independent fact checking.

Output limits are **not dollar spending limits**: input tokens and repeated manual
runs also cost money. Closing the browser does not guarantee cancellation of a paid
request already sent. Use OpenRouter's per-key credit limit; avoid automatic top-up
while evaluating this app.

### Low reasoning effort

For reasoning models, the app explicitly sends `reasoning.effort=low`, including
models where reasoning is mandatory. It omits reasoning parameters for non-reasoning
models. If a model publishes a list of supported efforts that excludes low, the model
check stops before any inference requests, including when selected as chairman.
Choose replacement models in `.env` and restart if the model check rejects your selection.
Low effort reduces the requested reasoning budget; it does not guarantee a final
answer within a small output-token cap.

An HTTP-successful request can still produce no visible answer: `finish_reason=length`
means the token limit was reached. Reasoning tokens count toward that limit and can
be billed even when final content is empty. The app reports this explicitly, retains
usage and costs, and does not treat reasoning text as a final answer. Nonempty answers
that hit the limit remain visible with a truncation warning. Raising
`MAX_OUTPUT_TOKENS` is a deliberate spending decision, never an automatic retry.
See [OpenRouter's reasoning token documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

## Local security and privacy

The supported entry point is `app.py` or `start.sh`. It deliberately:

- Binds to **127.0.0.1**, requires login, disables public sharing, and rejects foreign
  browser origins and unexpected Host headers.
- Keeps the API key on the Python server. HTTPS calls go to the fixed official
  `https://openrouter.ai/api/v1/chat/completions` endpoint, with certificate
  verification on, redirects off, and environment proxy settings ignored.
  Preflight uses the public `https://openrouter.ai/api/v1/models` catalog without
  a key or prompt. Catalog presence does not guarantee an eligible live ZDR endpoint;
  HTTP 404 may still indicate unavailable routing under the privacy requirements.
- Requires providers supporting the supplied parameters, disallows provider data
  collection, and requests zero data retention (`provider.zdr=true`). If no matching
  provider is available, the call fails; privacy requirements are not relaxed.
  OpenRouter may still allow safety-retention endpoints according to your account
  settings; select **ZDR only** in your OpenRouter privacy settings to exclude those.
- Displays all model responses as **plain text**, so response HTML, images and links
  cannot execute scripts, load remote images, or make automatic browser requests.
- Disables Gradio analytics, Hugging Face telemetry, browser run history, public API
  discovery for the council function, monitoring, MCP, and Node-based SSR.
- Blocks upload, file-serving, and proxy routes, denies file access across the
  filesystem, and limits incoming request bodies to 64 KiB.
- Writes no conversation files. Results are transient in browser/server memory;
  refresh clears the page, and stopping the process clears server memory. There is
  no conversation listing, export, upload, or persistence layer.
- Does not log prompts, response bodies, authorization headers, or raw provider
  exception messages. HTTP failures are logged using model ID and status only.

This is a **local single-user app**, not a hardened public deployment or a sandbox
against malware on your own computer. Treat the terminal's temporary password as
private. Your questions and responses still pass through OpenRouter and the selected
providers: ZDR is a provider policy constraint, not offline processing. Avoid sending
credentials or sensitive information. See [OpenRouter provider privacy](https://openrouter.ai/docs/guides/privacy/provider-logging)
and [key limits](https://openrouter.ai/docs/api_reference/authentication).

Do not expose this app with a tunnel, port forwarding, or a changed bind address.
Render/Docker deployment is intentionally deferred until its authentication,
networking, storage, and spending controls are designed.

## Reuse the council logic

```python
import asyncio
from backend.council import run_full_council

stage1, stage2, final, metadata = asyncio.run(
    run_full_council("What are the tradeoffs of this design?")
)
print(final["response"])
```

This makes real paid requests and applies the same workflow limits. For stage updates,
use the `iterate_council(question)` async generator instead.

```text
app.py                 Gradio UI and local launch controls
backend/config.py      Validated server-side settings
backend/openrouter.py  Bounded OpenRouter requests
backend/council.py     Independent answers, reviews, synthesis
tests/test_council.py  Mocked workflow and security regression checks
```

## Verify and update

Tests use mocked provider responses and spend no OpenRouter credits:

```bash
uv run --locked python -m unittest discover -s tests -v
```

Dependencies are pinned with hashes in `uv.lock`. For a deliberate dependency update,
run `uv sync --upgrade`, rerun the tests, and scan the new environment with `pip-audit`.
Gradio includes its prebuilt browser assets, so removing npm does not eliminate
third-party dependencies or the need for security updates.
