"""Tests for rich table rendering and report export of step metrics."""

import json

import pytest
from rich.table import Table

from openhands.sdk.conversation.metrics_report import (
    export_metrics_report,
    render_step_metrics_table,
)
from openhands.sdk.conversation.step_metrics import StepMetric
from openhands.sdk.llm.utils.metrics import Metrics


def _sample_breakdown() -> list[StepMetric]:
    return [
        StepMetric(
            step_index=0,
            llm_response_id="r0",
            tool_names=["terminal"],
            prompt_tokens=100,
            completion_tokens=10,
            cache_hit_rate=0.1,
            latency_seconds=0.5,
            cost_delta=0.01,
        ),
        StepMetric(
            step_index=1,
            llm_response_id="r1",
            tool_names=["file_editor"],
            prompt_tokens=200,
            completion_tokens=20,
            cache_hit_rate=0.2,
            latency_seconds=5.0,
            cost_delta=0.20,
        ),
    ]


def test_render_step_metrics_table_returns_table_with_one_row_per_step():
    metrics = Metrics(model_name="gpt-5")
    table = render_step_metrics_table(_sample_breakdown(), metrics)
    assert isinstance(table, Table)
    assert table.row_count == 2


def test_render_step_metrics_table_empty_breakdown_has_no_rows():
    table = render_step_metrics_table([], Metrics(model_name="gpt-5"))
    assert table.row_count == 0


def test_export_metrics_report_json_round_trips_steps_and_summary(tmp_path):
    metrics = Metrics(model_name="gpt-5")
    metrics.add_cost(0.21)
    out = tmp_path / "report.json"

    export_metrics_report(_sample_breakdown(), metrics, out, format="json")

    data = json.loads(out.read_text())
    assert len(data["steps"]) == 2
    assert data["steps"][1]["cost_delta"] == 0.20
    assert data["summary"]["accumulated_cost"] == pytest.approx(0.21)


def test_export_metrics_report_markdown_contains_top_costly_and_slow_steps(tmp_path):
    metrics = Metrics(model_name="gpt-5")
    out = tmp_path / "report.md"

    export_metrics_report(_sample_breakdown(), metrics, out, format="markdown")

    text = out.read_text()
    assert "r1" in text  # most costly and slowest step shows up
    assert "r0" in text  # timeline still includes every step


def test_export_metrics_report_markdown_is_utf8(tmp_path):
    # The report uses "—" (em dash) for missing values; on Windows the
    # default locale encoding (cp1252) can't round-trip it, so the file
    # must be written explicitly as UTF-8.
    metrics = Metrics(model_name="gpt-5")
    out = tmp_path / "report.md"

    export_metrics_report([], metrics, out, format="markdown")

    out.read_bytes().decode("utf-8")  # raises UnicodeDecodeError if not UTF-8


@pytest.mark.parametrize("format", ["markdown", "json"])
def test_export_metrics_report_empty_breakdown_does_not_crash(tmp_path, format):
    out = tmp_path / f"report.{format}"
    export_metrics_report([], Metrics(model_name="gpt-5"), out, format=format)
    assert out.exists()


def test_export_metrics_report_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        export_metrics_report(
            _sample_breakdown(),
            Metrics(model_name="gpt-5"),
            tmp_path / "x",
            format="yaml",  # type: ignore[arg-type]
        )
