"""
Driver plugin registry and built-in backends.

The driver layer has two responsibilities:
  1. Build the subprocess command for a backend CLI
  2. Parse backend-specific artifacts (events, session ids, last text)

Prompt construction and garden orchestration remain in system/driver.py.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Protocol

from .garden import read_garden_defaults


DEFAULT_DRIVER_ENV = "PAK2_DEFAULT_DRIVER"
DEFAULT_MODEL_ENV = "PAK2_DEFAULT_MODEL"
DEFAULT_REASONING_EFFORT_ENV = "PAK2_DEFAULT_REASONING_EFFORT"


class DriverPlugin(Protocol):
    name: str
    default_model: str
    default_reasoning_effort: str | None

    def build_launch_command(
        self,
        *,
        model: str,
        events_path: pathlib.Path,
        cwd: pathlib.Path | None,
        session_id: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        ...

    def build_reflection_command(
        self,
        *,
        model: str,
        reflection_path: pathlib.Path,
        session_id: str,
        cwd: pathlib.Path | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        ...

    def parse_events(self, events_path: pathlib.Path, returncode: int) -> tuple[dict, str]:
        ...

    def parse_session_id(self, events_path: pathlib.Path) -> str | None:
        ...

    def extract_last_text(self, path: pathlib.Path) -> str | None:
        ...


_REGISTRY: dict[str, DriverPlugin] = {}


def register_driver_plugin(plugin: DriverPlugin) -> None:
    _REGISTRY[plugin.name] = plugin


def get_driver_plugin(name: str) -> DriverPlugin:
    plugin = _REGISTRY.get(name)
    if plugin is None:
        raise KeyError(f"unknown driver plugin: {name}")
    return plugin


def list_driver_plugins() -> list[str]:
    return sorted(_REGISTRY)


def resolve_driver_name(goal: dict | None = None, *,
                        garden_root: pathlib.Path | None = None) -> str:
    if goal and goal.get("driver"):
        return str(goal["driver"])
    env_driver = os.environ.get(DEFAULT_DRIVER_ENV)
    if env_driver:
        return env_driver
    defaults = read_garden_defaults(garden_root=garden_root)
    if defaults.get("driver"):
        return str(defaults["driver"])
    return "codex"


def resolve_model_name(goal: dict | None = None, *, driver_name: str | None = None,
                       garden_root: pathlib.Path | None = None) -> str:
    if goal and goal.get("model"):
        return str(goal["model"])
    env_model = os.environ.get(DEFAULT_MODEL_ENV)
    if env_model:
        return env_model
    defaults = read_garden_defaults(garden_root=garden_root)
    if defaults.get("model"):
        return str(defaults["model"])
    plugin = get_driver_plugin(
        driver_name or resolve_driver_name(goal, garden_root=garden_root)
    )
    return plugin.default_model


def resolve_reasoning_effort(goal: dict | None = None, *,
                             garden_root: pathlib.Path | None = None) -> str | None:
    if goal and goal.get("reasoning_effort"):
        return str(goal["reasoning_effort"])
    env_effort = os.environ.get(DEFAULT_REASONING_EFFORT_ENV)
    if env_effort:
        return env_effort
    defaults = read_garden_defaults(garden_root=garden_root)
    if defaults.get("reasoning_effort"):
        return str(defaults["reasoning_effort"])
    plugin = get_driver_plugin(resolve_driver_name(goal, garden_root=garden_root))
    return plugin.default_reasoning_effort


def _read_json_lines(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _read_text_if_present(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return content or None


@dataclass(frozen=True)
class ClaudeDriverPlugin:
    name: str = "claude"
    default_model: str = "claude-opus-4-6"
    default_reasoning_effort: str | None = None

    def build_launch_command(
        self,
        *,
        model: str,
        events_path: pathlib.Path,
        cwd: pathlib.Path | None,
        session_id: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        command = [
            "claude",
            "--model", model,
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", "acceptEdits",
        ]
        if session_id:
            command += ["--resume", session_id]
        return command

    def build_reflection_command(
        self,
        *,
        model: str,
        reflection_path: pathlib.Path,
        session_id: str,
        cwd: pathlib.Path | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        return [
            "claude",
            "--model", model,
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", "acceptEdits",
            "--resume", session_id,
        ]

    def parse_events(self, events_path: pathlib.Path, returncode: int) -> tuple[dict, str]:
        cost = {"source": "unknown"}
        status = "success" if returncode == 0 else "failure"

        result_event = None
        for event in _read_json_lines(events_path):
            if event.get("type") == "result":
                result_event = event

        if not result_event:
            return cost, status

        usage = result_event.get("usage", {})
        total_cost = result_event.get("total_cost_usd")
        cost = {
            "source": "provider" if total_cost is not None else "estimated",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        }
        if total_cost is not None:
            cost["actual_usd"] = total_cost

        subtype = result_event.get("subtype", "")
        if subtype == "success":
            status = "success"
        elif subtype == "error_max_turns":
            status = "failure"
        elif returncode != 0:
            status = "failure"

        return cost, status

    def parse_session_id(self, events_path: pathlib.Path) -> str | None:
        for event in _read_json_lines(events_path):
            if event.get("type") == "result":
                return event.get("session_id")
        return None

    def extract_last_text(self, path: pathlib.Path) -> str | None:
        last_text = None
        for event in _read_json_lines(path):
            if event.get("type") == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        last_text = block["text"]
            elif event.get("type") == "result" and event.get("result"):
                last_text = event["result"]
        return last_text if last_text and str(last_text).strip() else None


@dataclass(frozen=True)
class CodexDriverPlugin:
    name: str = "codex"
    default_model: str = "gpt-5.4"
    default_reasoning_effort: str | None = "xhigh"

    def build_launch_command(
        self,
        *,
        model: str,
        events_path: pathlib.Path,
        cwd: pathlib.Path | None,
        session_id: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        last_message = self._last_message_path(events_path)
        command = [
            "codex",
            "exec",
        ]
        if session_id:
            command += ["resume"]
        command += [
            "--json",
            "--model", model,
            "--output-last-message", str(last_message),
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if reasoning_effort:
            command += ["-c", f'model_reasoning_effort="{reasoning_effort}"']
        if session_id:
            command.append(session_id)
        command.append("-")
        return command

    def build_reflection_command(
        self,
        *,
        model: str,
        reflection_path: pathlib.Path,
        session_id: str,
        cwd: pathlib.Path | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        last_message = reflection_path.with_name("reflection.md")
        command = [
            "codex",
            "exec",
            "resume",
            "--json",
            "--model", model,
            "--output-last-message", str(last_message),
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if reasoning_effort:
            command += ["-c", f'model_reasoning_effort="{reasoning_effort}"']
        command += [session_id, "-"]
        return command

    def parse_events(self, events_path: pathlib.Path, returncode: int) -> tuple[dict, str]:
        status = "success" if returncode == 0 else "failure"
        cost = {"source": "unknown"}
        usage = None
        for event in _read_json_lines(events_path):
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
        if usage:
            cost = {
                "source": "provider",
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_tokens": usage.get("cached_input_tokens", 0),
            }
        return cost, status

    def parse_session_id(self, events_path: pathlib.Path) -> str | None:
        for event in _read_json_lines(events_path):
            if event.get("type") == "thread.started":
                return event.get("thread_id")
        return None

    def extract_last_text(self, path: pathlib.Path) -> str | None:
        last_message = self._last_message_path(path)
        return _read_text_if_present(last_message)

    @staticmethod
    def _last_message_path(path: pathlib.Path) -> pathlib.Path:
        if path.name == "reflection.jsonl":
            return path.with_name("reflection.md")
        return path.with_name("last-message.md")


register_driver_plugin(ClaudeDriverPlugin())
register_driver_plugin(CodexDriverPlugin())
