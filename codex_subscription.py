#!/usr/bin/env python3
"""Small helper for structured, batched calls through a ChatGPT Codex login."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CODEX_PATH = Path("/Applications/Codex.app/Contents/Resources/codex")


@dataclass
class CodexBatchResult:
    payload: dict
    latency_seconds: float
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_output_tokens: int | None


class CodexBatchClient:
    def __init__(
        self,
        model: str,
        reasoning_effort: str,
        timeout: int = 900,
        max_retries: int = 2,
        codex_path: str | None = None,
    ):
        discovered = codex_path or shutil.which("codex")
        if discovered is None and DEFAULT_CODEX_PATH.exists():
            discovered = str(DEFAULT_CODEX_PATH)
        if discovered is None:
            raise FileNotFoundError("Could not find the Codex CLI.")
        self.codex_path = discovered
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries

    def ask(self, prompt: str, schema: dict) -> CodexBatchResult:
        with tempfile.TemporaryDirectory(prefix="ece175b-codex-") as directory:
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command = [
                self.codex_path,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                self.model,
                "--config",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "--output-schema",
                str(schema_path),
                "--json",
                "-",
            ]
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                start = time.perf_counter()
                try:
                    process = subprocess.run(
                        command,
                        input=prompt,
                        text=True,
                        capture_output=True,
                        timeout=self.timeout,
                        check=True,
                    )
                    result = self._parse_events(process.stdout)
                    result.latency_seconds = time.perf_counter() - start
                    return result
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    last_error = exc
                    if attempt == self.max_retries:
                        stderr = getattr(exc, "stderr", "") or ""
                        raise RuntimeError(
                            f"Codex batch failed after {attempt + 1} attempts: "
                            f"{exc}\n{stderr[-1000:]}"
                        ) from exc
                    time.sleep(2**attempt)
            raise RuntimeError(f"Codex batch failed: {last_error}")

    @staticmethod
    def _parse_events(stdout: str) -> CodexBatchResult:
        message: str | None = None
        usage: dict = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            event = json.loads(line)
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    message = item.get("text")
            elif event.get("type") == "turn.completed":
                usage = event.get("usage", {})
        if message is None:
            raise ValueError("Codex JSONL contained no final agent message.")
        return CodexBatchResult(
            payload=json.loads(message),
            latency_seconds=0.0,
            input_tokens=_optional_int(usage.get("input_tokens")),
            cached_input_tokens=_optional_int(usage.get("cached_input_tokens")),
            output_tokens=_optional_int(usage.get("output_tokens")),
            reasoning_output_tokens=_optional_int(usage.get("reasoning_output_tokens")),
        )


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def answer_batch_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    },
                    "required": ["id", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def grounding_batch_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "supported": {"type": "boolean"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "supporting_quote": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "supported",
                        "confidence",
                        "supporting_quote",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }
