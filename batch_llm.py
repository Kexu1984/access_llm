#!/usr/bin/env python3
"""Batch LLM caller (OpenAI-compatible).

Designed for SiliconFlow/OpenAI-style APIs:
- Endpoint: POST {base_url}/chat/completions
- Header: Authorization: Bearer <API_KEY>

Features:
- API key via env var or CLI
- Batch input from .txt or .jsonl
- Concurrency + retries with exponential backoff
- Writes results to JSONL

This is a prototype; adjust base_url/model to your deployment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx


ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def load_simple_dotenv(dotenv_path: str = ".env") -> None:
    """Load KEY=VALUE lines from a local .env into os.environ (if not already set).

    This avoids extra dependencies. It is intentionally minimal.
    """

    if not os.path.exists(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or not ENV_VAR_NAME_RE.match(key):
                    continue

                if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
                    value = value[1:-1]

                os.environ.setdefault(key, value)
    except Exception:
        # Never fail hard due to dotenv parsing.
        return


# Load local .env early so defaults can see it.
load_simple_dotenv()


DEFAULT_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_MODEL = os.environ.get("SILICONFLOW_MODEL", "")
DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"


@dataclass(frozen=True)
class WorkItem:
    id: str
    messages: List[Dict[str, Any]]
    meta: Dict[str, Any]


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def build_endpoint(base_url: str, endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    base = base_url.rstrip("/")
    ep = endpoint.lstrip("/")
    return f"{base}/{ep}"


def load_items(path: str, system_prompt: str) -> List[WorkItem]:
    if path.lower().endswith(".jsonl"):
        return load_items_jsonl(path, system_prompt)
    return load_items_txt(path, system_prompt)


def load_items_txt(path: str, system_prompt: str) -> List[WorkItem]:
    items: List[WorkItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            prompt = line.strip("\n").strip()
            if not prompt:
                continue
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            items.append(WorkItem(id=str(i), messages=messages, meta={"prompt": prompt}))
    return items


def load_items_jsonl(path: str, system_prompt: str) -> List[WorkItem]:
    items: List[WorkItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            item_id = str(row.get("id") or i)

            if "messages" in row and isinstance(row["messages"], list):
                messages = row["messages"]
            else:
                prompt = row.get("prompt")
                if prompt is None:
                    raise ValueError(
                        f"JSONL line {i} must contain either 'messages' (list) or 'prompt'."
                    )
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": str(prompt)})

            meta = {k: v for k, v in row.items() if k not in {"messages"}}
            items.append(WorkItem(id=item_id, messages=messages, meta=meta))
    return items


def extract_text(resp_json: Dict[str, Any]) -> str:
    # OpenAI-style: choices[0].message.content
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if content is not None:
                return str(content)
        # Some APIs might return choices[0].text
        text = choices[0].get("text")
        if text is not None:
            return str(text)
    return ""


def is_retryable_status(status_code: int) -> bool:
    return status_code in (408, 429, 500, 502, 503, 504)


def build_timeout(read_timeout_s: float) -> httpx.Timeout:
    # Keep connect/write/pool bounded so failures surface quickly.
    # Treat the CLI --timeout-s as the read timeout (model generation time).
    return httpx.Timeout(connect=10.0, read=read_timeout_s, write=30.0, pool=10.0)


async def call_one(
    client: httpx.AsyncClient,
    endpoint_url: str,
    api_key: str,
    model: str,
    item: WorkItem,
    temperature: float,
    max_tokens: Optional[int],
    top_p: Optional[float],
    extra: Dict[str, Any],
    timeout_s: float,
    max_retries: int,
    base_backoff_s: float,
    semaphore: asyncio.Semaphore,
    mock: bool,
) -> Dict[str, Any]:
    async with semaphore:
        started = time.time()

        if mock:
            await asyncio.sleep(0.05)
            return {
                "id": item.id,
                "ok": True,
                "output": f"[MOCK] {item.meta.get('prompt') or extract_text({'choices': [{'message': {'content': ''}}]})}",
                "latency_s": round(time.time() - started, 4),
                "meta": item.meta,
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": model,
            "messages": item.messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        payload.update(extra)

        attempt = 0
        last_error: Optional[str] = None
        last_status: Optional[int] = None

        while True:
            attempt += 1
            try:
                r = await client.post(
                    endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=build_timeout(timeout_s),
                )

                last_status = r.status_code
                if r.status_code >= 400:
                    # Try to parse error body for diagnostics
                    try:
                        err_json = r.json()
                        last_error = _json_dumps(err_json)
                    except Exception:
                        last_error = r.text[:2000]

                    if attempt <= max_retries and is_retryable_status(r.status_code):
                        sleep_s = base_backoff_s * (2 ** (attempt - 1))
                        sleep_s += random.uniform(0, min(0.25, sleep_s * 0.1))
                        await asyncio.sleep(sleep_s)
                        continue

                    return {
                        "id": item.id,
                        "ok": False,
                        "status": r.status_code,
                        "error": last_error or "HTTP error",
                        "latency_s": round(time.time() - started, 4),
                        "meta": item.meta,
                    }

                resp_json = r.json()
                text = extract_text(resp_json)
                usage = resp_json.get("usage")

                return {
                    "id": item.id,
                    "ok": True,
                    "output": text,
                    "usage": usage,
                    "latency_s": round(time.time() - started, 4),
                    "meta": item.meta,
                }

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt <= max_retries:
                    sleep_s = base_backoff_s * (2 ** (attempt - 1))
                    sleep_s += random.uniform(0, min(0.25, sleep_s * 0.1))
                    await asyncio.sleep(sleep_s)
                    continue
                return {
                    "id": item.id,
                    "ok": False,
                    "status": last_status,
                    "error": last_error,
                    "latency_s": round(time.time() - started, 4),
                    "meta": item.meta,
                }
            except Exception as e:
                return {
                    "id": item.id,
                    "ok": False,
                    "status": last_status,
                    "error": f"{type(e).__name__}: {e}",
                    "latency_s": round(time.time() - started, 4),
                    "meta": item.meta,
                }


async def run_batch(args: argparse.Namespace) -> int:
    api_key_env = (args.api_key_env or DEFAULT_API_KEY_ENV).strip()
    if not ENV_VAR_NAME_RE.match(api_key_env):
        # Avoid leaking accidental secrets (e.g. pasting a real key into --api-key-env)
        api_key_env = DEFAULT_API_KEY_ENV

    api_key = (args.api_key or "").strip() or os.environ.get(api_key_env, "").strip()
    if not api_key and not args.mock:
        print(
            f"Missing API key. Set env {api_key_env} or pass --api-key. "
            "(Or use --mock for a dry run.)",
            file=sys.stderr,
        )
        return 2

    model = args.model or DEFAULT_MODEL
    if not model and not args.mock:
        print("Missing model. Pass --model or set SILICONFLOW_MODEL.", file=sys.stderr)
        return 2

    endpoint_url = build_endpoint(args.base_url, args.endpoint)

    extra: Dict[str, Any] = {}
    if args.extra_json:
        extra.update(json.loads(args.extra_json))

    items = load_items(args.input, args.system)
    if not items:
        print("No valid inputs found.", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    limits = httpx.Limits(
        max_connections=max(args.concurrency * 2, 10),
        max_keepalive_connections=max(args.concurrency, 5),
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            call_one(
                client=client,
                endpoint_url=endpoint_url,
                api_key=api_key,
                model=model or "mock-model",
                item=item,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                extra=extra,
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
                base_backoff_s=args.base_backoff_s,
                semaphore=semaphore,
                mock=args.mock,
            )
            for item in items
        ]

        ok_count = 0
        fail_count = 0

        started = time.time()
        with open(args.output, "w", encoding="utf-8") as out:
            for fut in asyncio.as_completed(tasks):
                result = await fut
                if result.get("ok"):
                    ok_count += 1
                else:
                    fail_count += 1
                out.write(_json_dumps(result) + "\n")

                if args.progress:
                    done = ok_count + fail_count
                    total = len(items)
                    print(f"done {done}/{total} ok={ok_count} fail={fail_count}", file=sys.stderr)

        elapsed = time.time() - started
        print(
            f"Finished: total={len(items)} ok={ok_count} fail={fail_count} "
            f"elapsed_s={elapsed:.2f} output={args.output}",
            file=sys.stderr,
        )

    return 0 if fail_count == 0 else 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch call LLM API (OpenAI-compatible).")

    p.add_argument("--input", required=True, help="Input .txt (one prompt per line) or .jsonl")
    p.add_argument("--output", required=True, help="Output .jsonl")

    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--endpoint", default="chat/completions", help="Endpoint path or full URL")

    p.add_argument("--api-key", default="", help="API key (recommended: use env)")
    p.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Env var name for API key")

    p.add_argument("--model", default="", help="Model name (or set SILICONFLOW_MODEL)")
    p.add_argument("--system", default="", help="Optional system prompt")

    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)

    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--base-backoff-s", type=float, default=1.0)

    p.add_argument(
        "--extra-json",
        default="",
        help="Extra JSON merged into request payload, e.g. '{""seed"":123}'",
    )

    p.add_argument("--progress", action="store_true", help="Print progress to stderr")
    p.add_argument("--mock", action="store_true", help="Do not call network; return mock outputs")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run_batch(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
