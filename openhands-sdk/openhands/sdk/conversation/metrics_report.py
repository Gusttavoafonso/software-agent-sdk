"""Rich table rendering and Markdown/JSON export for step-level telemetry."""

import json
from pathlib import Path
from typing import Literal

from rich.table import Table

from openhands.sdk.conversation.step_metrics import StepMetric
from openhands.sdk.llm.utils.metrics import Metrics


_COLUMNS = ("Step", "Tools", "Prompt", "Completion", "Cache Hit", "Latency (s)", "Cost")


def _format_cache_hit_rate(cache_hit_rate: float | None) -> str:
    return "—" if cache_hit_rate is None else f"{cache_hit_rate:.0%}"


def _format_latency(latency_seconds: float | None) -> str:
    return "—" if latency_seconds is None else f"{latency_seconds:.2f}"


def _format_cost(cost_delta: float | None) -> str:
    return "—" if cost_delta is None else f"${cost_delta:.4f}"


def _step_row(step: StepMetric) -> tuple[str, ...]:
    return (
        str(step.step_index),
        ", ".join(step.tool_names),
        "—" if step.prompt_tokens is None else str(step.prompt_tokens),
        "—" if step.completion_tokens is None else str(step.completion_tokens),
        _format_cache_hit_rate(step.cache_hit_rate),
        _format_latency(step.latency_seconds),
        _format_cost(step.cost_delta),
    )


def render_step_metrics_table(breakdown: list[StepMetric], metrics: Metrics) -> Table:
    snapshot = metrics.get_snapshot()
    caption = (
        f"Total cost: ${snapshot.accumulated_cost:.4f} | "
        f"Cache hit rate: {_format_cache_hit_rate(snapshot.cache_hit_rate)}"
    )
    table = Table(title="Step-Level Telemetry", caption=caption)
    for column in _COLUMNS:
        table.add_column(column)
    for step in breakdown:
        table.add_row(*_step_row(step))
    return table


def export_metrics_report(
    breakdown: list[StepMetric],
    metrics: Metrics,
    path: str | Path,
    format: Literal["markdown", "json"] = "markdown",
) -> None:
    if format == "json":
        content = json.dumps(
            {
                "summary": metrics.get_snapshot().model_dump(),
                "steps": [step.model_dump() for step in breakdown],
            },
            indent=2,
        )
    elif format == "markdown":
        content = _render_markdown_report(breakdown, metrics)
    else:
        raise ValueError(f"Unsupported format: {format!r}")
    Path(path).write_text(content)


def _render_markdown_report(breakdown: list[StepMetric], metrics: Metrics) -> str:
    snapshot = metrics.get_snapshot()
    lines = [
        "# Step-Level Telemetry Report",
        "",
        "## Summary",
        f"- Accumulated cost: ${snapshot.accumulated_cost:.4f}",
        f"- Cache hit rate: {_format_cache_hit_rate(snapshot.cache_hit_rate)}",
        f"- Total steps: {len(breakdown)}",
        "",
        "## Top 5 Most Costly Steps",
        *_top_steps_lines(breakdown, key=lambda step: step.cost_delta),
        "",
        "## Top 5 Slowest Steps",
        *_top_steps_lines(breakdown, key=lambda step: step.latency_seconds),
        "",
        "## Timeline",
        f"| {' | '.join(_COLUMNS)} |",
        f"|{'|'.join(['---'] * len(_COLUMNS))}|",
        *(f"| {' | '.join(_step_row(step))} |" for step in breakdown),
    ]
    return "\n".join(lines)


def _top_steps_lines(breakdown: list[StepMetric], *, key) -> list[str]:
    ranked = sorted(
        (step for step in breakdown if key(step) is not None),
        key=key,
        reverse=True,
    )[:5]
    if not ranked:
        return ["- (none)"]
    return [
        f"- Step {step.step_index} [{step.llm_response_id}] "
        f"({', '.join(step.tool_names)}): "
        f"cost={_format_cost(step.cost_delta)}, "
        f"latency={_format_latency(step.latency_seconds)}s"
        for step in ranked
    ]
