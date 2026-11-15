"""Bounded OpenRouter requests, model preflight, and safe billing diagnostics."""

import asyncio
import json
import logging
import math

import httpx

from .config import (
    MAX_HTTP_RESPONSE_BYTES, MAX_RESPONSE_CHARS, OPENROUTER_API_URL,
    OPENROUTER_MODELS_URL, REQUEST_TIMEOUT_SECONDS, get_settings,
)

logger = logging.getLogger(__name__)


class ModelCheckError(RuntimeError):
    """A safe preflight failure. No generation request has been made."""


def low_reasoning_config(model_info: dict) -> dict | None:
    """Request low effort for reasoning models, including mandatory-reasoning models."""
    supported = model_info.get("supported_parameters")
    if not isinstance(supported, list):
        raise ModelCheckError("Model capability information is unavailable.")
    reasoning = model_info.get("reasoning")
    if reasoning is None and not {"reasoning", "reasoning_effort"}.intersection(supported):
        return None  # Non-reasoning model: do not send unsupported parameters.
    efforts = reasoning.get("supported_efforts") if isinstance(reasoning, dict) else None
    if isinstance(efforts, list) and "low" not in efforts:
        raise ModelCheckError("This model does not advertise support for low reasoning effort.")
    return {"effort": "low"}


async def _read_json(response: httpx.Response, max_bytes: int) -> dict:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ValueError("Oversized response")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("Invalid response")
    return data


async def validate_model_configuration() -> dict[str, dict]:
    """Look up public model metadata only. This makes no inference request."""
    settings = get_settings()
    try:
        async with asyncio.timeout(20):
            async with httpx.AsyncClient(timeout=20, trust_env=False, follow_redirects=False) as client:
                async with client.stream("GET", OPENROUTER_MODELS_URL) as response:
                    response.raise_for_status()
                    data = await _read_json(response, 8_000_000)
        catalog = {row["id"]: row for row in data["data"] if isinstance(row, dict) and isinstance(row.get("id"), str)}
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
        raise ModelCheckError("Could not check the OpenRouter model catalog. No paid requests were sent.") from None
    problems = []
    selected = {}
    for model in dict.fromkeys((*settings.models, settings.chairman)):
        if model not in catalog:
            problems.append(f"{model}: not listed in the current OpenRouter catalog; check the model ID.")
            continue
        try:
            low_reasoning_config(catalog[model])
        except ModelCheckError as exc:
            problems.append(f"{model}: {exc}")
        selected[model] = catalog[model]
    if problems:
        raise ModelCheckError("Model check stopped the run before any paid requests:\n" + "\n".join(problems))
    return selected


def _usage(data: dict) -> dict:
    """Whitelist numeric fields; never copy arbitrary provider metadata to the UI."""
    raw = data.get("usage")
    if not isinstance(raw, dict):
        return {}
    usage = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        value = raw.get(name)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            usage[name] = value
    details = raw.get("completion_tokens_details")
    value = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if type(value) is int and value >= 0:
        usage["reasoning_tokens"] = value
    return usage


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part["text"] for part in content if isinstance(part, dict)
                         and part.get("type") in {"text", "output_text"} and isinstance(part.get("text"), str))
    return ""


def _failure(message: str, *, usage=None, finish_reason=None) -> dict:
    return {"content": None, "error": message, "usage": usage or {},
            "finish_reason": finish_reason, "warnings": []}


def _http_failure(status: int) -> dict:
    explanations = {
        400: "The model rejected the request parameters.",
        401: "OpenRouter rejected the API key.",
        402: "The key has insufficient credits or reached its spending limit.",
        403: "The request was forbidden by OpenRouter or provider policy.",
        404: "No endpoint was available for this model and the current privacy/parameter requirements. Check the model ID and ZDR availability.",
        429: "OpenRouter or the provider rate-limited this request.",
    }
    return _failure(f"HTTP {status}: {explanations.get(status, 'The upstream request failed.')} No retry was made.")


async def query_model(model: str, messages: list[dict[str, str]], *, model_info: dict | None = None) -> dict:
    settings = get_settings()
    if not settings.api_key:
        raise ValueError("Set OPENROUTER_API_KEY in .env before running the council.")
    if model not in (*settings.models, settings.chairman):
        raise ValueError("The requested model is not configured for this council.")
    # Direct Python calls get the same preflight as the full council workflow.
    if model_info is None:
        model_info = (await validate_model_configuration())[model]
    payload = {
        "model": model, "messages": messages, "stream": False,
        "max_tokens": settings.max_output_tokens,
        "provider": {"data_collection": "deny", "zdr": True, "require_parameters": True},
    }
    reasoning = low_reasoning_config(model_info)
    if reasoning is not None:
        payload["reasoning"] = reasoning
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10),
                trust_env=False, follow_redirects=False,
            ) as client:
                async with client.stream(
                    "POST", OPENROUTER_API_URL,
                    headers={"Authorization": f"Bearer {settings.api_key}"}, json=payload,
                ) as response:
                    if response.status_code != 200:
                        result = _http_failure(response.status_code)
                        logger.warning("Model %s: %s", model, result["error"])
                        return result
                    if "text/event-stream" in response.headers.get("content-type", ""):
                        return _failure("The provider returned a stream despite stream=false. No retry was made; the request may have been billed.")
                    data = await _read_json(response, MAX_HTTP_RESPONSE_BYTES)
        usage = _usage(data)
        if "error" in data:
            return _failure("OpenRouter returned an error inside its response. No retry was made; check the OpenRouter activity log.", usage=usage)
        choice = data["choices"][0]
        message = choice["message"]
        finish = choice.get("finish_reason")
        finish = finish if finish in {"stop", "length", "tool_calls", "content_filter", "error"} else None
        content = _text_content(message.get("content"))
        if len(content) > MAX_RESPONSE_CHARS:
            return _failure("The model response exceeded the app's display limit. The request may have been billed.", usage=usage, finish_reason=finish)
        if not content.strip():
            if finish == "length":
                error = (f"No final answer: the {settings.max_output_tokens:,}-token output limit was reached. "
                         "Reasoning can consume this limit before visible text; generated tokens are still billed. "
                         "No retry was made. Increasing MAX_OUTPUT_TOKENS explicitly may help, but can increase cost.")
            elif message.get("reasoning") or message.get("reasoning_content") or message.get("reasoning_details"):
                error = "The provider returned reasoning only, with no final answer. Low reasoning effort does not guarantee visible output within the token cap. The request may have been billed. No retry was made."
            else:
                error = "The provider returned no final answer. The request may have been billed. See token usage and the OpenRouter activity log. No retry was made."
            result = _failure(error, usage=usage, finish_reason=finish)
            logger.warning("Model %s: %s", model, error)
            return result
        warnings = []
        if finish == "length":
            warnings.append("The token limit was reached; this answer is incomplete.")
        return {"content": content, "error": None, "usage": usage,
                "finish_reason": finish, "warnings": warnings}
    except (httpx.HTTPError, TimeoutError):
        error = "A network error or timeout interrupted the request. It may have been billed. No retry was made."
    except (ValueError, KeyError, IndexError, TypeError):
        error = "The provider response was malformed or exceeded the size limit. It may have been billed. No retry was made."
    logger.warning("Model %s: %s", model, error)
    return _failure(error)


async def query_models_parallel(models: tuple[str, ...], messages: list[dict[str, str]], *, catalog=None) -> dict:
    if catalog is None:
        catalog = await validate_model_configuration()
    responses = await asyncio.gather(*(query_model(model, messages, model_info=catalog[model]) for model in models))
    return dict(zip(models, responses))
