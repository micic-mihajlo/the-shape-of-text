#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.generate_social_posts import generation_record, prompt_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chat_payload(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You rewrite rough founder notes into direct, specific social posts. "
                    "Return only the final post."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }


def post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(url: str, *, timeout: float, poll_interval: float) -> None:
    deadline = time.time() + timeout
    payload = {
        "model": "healthcheck",
        "messages": [{"role": "user", "content": "Say ready."}],
        "max_tokens": 4,
        "temperature": 0,
    }
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            post_json(url, payload, timeout=10)
            return
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            time.sleep(poll_interval)
    raise TimeoutError(f"llama.cpp server did not become ready: {last_error}")


def completion_from_response(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected chat completion response: {response!r}") from exc
    if not isinstance(content, str):
        raise ValueError(f"unexpected completion content: {content!r}")
    return content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate founder/social eval posts through llama.cpp OpenAI-compatible chat."
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="OpenAI-compatible chat completions endpoint exposed by llama-server.",
    )
    parser.add_argument("--model", default="diffusiongemma")
    parser.add_argument(
        "--briefs-file",
        type=Path,
        default=Path("configs/founder_rewrite_eval_briefs.jsonl"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/diffusiongemma_posts.jsonl"),
    )
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--server-timeout", type=float, default=900)
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_wait:
        wait_for_server(
            args.endpoint,
            timeout=args.server_timeout,
            poll_interval=args.poll_interval,
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for brief in read_jsonl(args.briefs_file):
            prompt = prompt_text(brief)
            try:
                response = post_json(
                    args.endpoint,
                    chat_payload(prompt, args),
                    timeout=args.request_timeout,
                )
            except URLError as exc:
                raise RuntimeError(f"request to {args.endpoint} failed") from exc
            completion = completion_from_response(response)
            record = generation_record(brief, completion)
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(f"generated {record['id']}", flush=True)


if __name__ == "__main__":
    main()
