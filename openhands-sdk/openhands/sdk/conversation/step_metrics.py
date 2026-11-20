"""Per-step telemetry derived from an event timeline and an LLM's Metrics.

See ``build_step_breakdown`` for the correlation rule: one LLM response
(grouped by ``llm_response_id``) is one step, matched against the main agent
LLM's ``Metrics`` ledger — never the combined condenser/critic ledger, so
step costs reconcile exactly against that ledger's ``accumulated_cost``.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from openhands.sdk.event.base import Event
from openhands.sdk.event.llm_convertible.action import ActionEvent
from openhands.sdk.llm.utils.metrics import (
    Cost,
    Metrics,
    ResponseLatency,
    TokenUsage,
    compute_cache_hit_rate,
)


class StepMetric(BaseModel):
    step_index: int
    llm_response_id: str
    tool_names: list[str]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit_rate: float | None = None
    latency_seconds: float | None = None
    cost_delta: float | None = None


def build_step_breakdown(events: Sequence[Event], metrics: Metrics) -> list[StepMetric]:
    """Correlate ``ActionEvent``s with ``metrics`` by ``llm_response_id``.

    All ``ActionEvent``s sharing one ``llm_response_id`` (parallel tool
    calls from a single LLM response) collapse into one step, so cost and
    latency are never double-counted.
    """
    tool_names_by_response_id: dict[str, list[str]] = {}
    for event in events:
        if not isinstance(event, ActionEvent):
            continue
        tool_names_by_response_id.setdefault(event.llm_response_id, []).append(
            event.tool_name
        )

    costs_by_response_id: dict[str, Cost] = {}
    for cost in metrics.costs:
        costs_by_response_id.setdefault(cost.response_id, cost)

    latencies_by_response_id: dict[str, ResponseLatency] = {}
    for latency in metrics.response_latencies:
        latencies_by_response_id.setdefault(latency.response_id, latency)

    usages_by_response_id: dict[str, TokenUsage] = {}
    for usage in metrics.token_usages:
        usages_by_response_id.setdefault(usage.response_id, usage)

    breakdown: list[StepMetric] = []
    for step_index, (response_id, tool_names) in enumerate(
        tool_names_by_response_id.items()
    ):
        cost = costs_by_response_id.get(response_id)
        latency = latencies_by_response_id.get(response_id)
        usage = usages_by_response_id.get(response_id)
        breakdown.append(
            StepMetric(
                step_index=step_index,
                llm_response_id=response_id,
                tool_names=tool_names,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                cache_hit_rate=(
                    compute_cache_hit_rate(usage.prompt_tokens, usage.cache_read_tokens)
                    if usage
                    else None
                ),
                latency_seconds=latency.latency if latency else None,
                cost_delta=cost.cost if cost else None,
            )
        )
    return breakdown
