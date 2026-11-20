"""Tests for per-step telemetry correlation."""

from openhands.sdk.conversation.step_metrics import (
    StepMetric,
    build_step_breakdown,
)
from openhands.sdk.event.llm_convertible.action import ActionEvent
from openhands.sdk.event.llm_convertible.message import MessageEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.llm.utils.metrics import Metrics


def _action_event(*, response_id: str, tool_name: str, call_id: str) -> ActionEvent:
    # `action=None` is the "non-executable" shape the SDK itself uses (see
    # ActionEvent.action's docstring); build_step_breakdown only reads
    # llm_response_id/tool_name, so every test below exercises that shape.
    return ActionEvent(
        source="agent",
        thought=[TextContent(text="thinking")],
        action=None,
        tool_name=tool_name,
        tool_call_id=call_id,
        tool_call=MessageToolCall(
            id=call_id, name=tool_name, arguments="{}", origin="completion"
        ),
        llm_response_id=response_id,
    )


def test_build_step_breakdown_empty_events_returns_empty_list():
    assert build_step_breakdown([], Metrics(model_name="gpt-5")) == []


def test_build_step_breakdown_single_action_matches_metrics():
    metrics = Metrics(model_name="gpt-5")
    metrics.add_cost(0.05, response_id="resp-1")
    metrics.add_response_latency(1.25, "resp-1")
    metrics.add_token_usage(
        prompt_tokens=100,
        completion_tokens=20,
        cache_read_tokens=25,
        cache_write_tokens=0,
        context_window=4096,
        response_id="resp-1",
    )
    events = [_action_event(response_id="resp-1", tool_name="terminal", call_id="c1")]

    breakdown = build_step_breakdown(events, metrics)

    assert breakdown == [
        StepMetric(
            step_index=0,
            llm_response_id="resp-1",
            tool_names=["terminal"],
            prompt_tokens=100,
            completion_tokens=20,
            cache_hit_rate=0.25,
            latency_seconds=1.25,
            cost_delta=0.05,
        )
    ]


def test_build_step_breakdown_groups_parallel_tool_calls_into_one_step():
    metrics = Metrics(model_name="gpt-5")
    metrics.add_cost(0.10, response_id="resp-1")
    metrics.add_response_latency(2.0, "resp-1")
    events = [
        _action_event(response_id="resp-1", tool_name="terminal", call_id="c1"),
        _action_event(response_id="resp-1", tool_name="file_editor", call_id="c2"),
    ]

    breakdown = build_step_breakdown(events, metrics)

    assert len(breakdown) == 1
    assert breakdown[0].tool_names == ["terminal", "file_editor"]
    assert breakdown[0].cost_delta == 0.10  # not 0.20 — no double counting


def test_build_step_breakdown_non_executable_action_still_produces_step():
    # action=None (the SDK's "non-executable" shape) must still surface as a
    # step: build_step_breakdown keys off llm_response_id/tool_name, not
    # whether the action itself ran.
    metrics = Metrics(model_name="gpt-5")
    metrics.add_cost(0.03, response_id="resp-1")
    events = [_action_event(response_id="resp-1", tool_name="terminal", call_id="c1")]

    breakdown = build_step_breakdown(events, metrics)

    assert len(breakdown) == 1
    assert breakdown[0].cost_delta == 0.03


def test_build_step_breakdown_unpriced_response_has_none_cost():
    metrics = Metrics(model_name="gpt-5")
    metrics.add_response_latency(0.5, "resp-1")  # no add_cost call
    events = [_action_event(response_id="resp-1", tool_name="terminal", call_id="c1")]

    breakdown = build_step_breakdown(events, metrics)

    assert breakdown[0].cost_delta is None
    assert breakdown[0].latency_seconds == 0.5


def test_build_step_breakdown_two_steps_in_order():
    metrics = Metrics(model_name="gpt-5")
    metrics.add_cost(0.01, response_id="resp-1")
    metrics.add_cost(0.02, response_id="resp-2")
    events = [
        _action_event(response_id="resp-1", tool_name="terminal", call_id="c1"),
        _action_event(response_id="resp-2", tool_name="file_editor", call_id="c2"),
    ]

    breakdown = build_step_breakdown(events, metrics)

    assert [s.step_index for s in breakdown] == [0, 1]
    assert [s.cost_delta for s in breakdown] == [0.01, 0.02]


def test_build_step_breakdown_ignores_non_action_events():
    metrics = Metrics(model_name="gpt-5")
    message_event = MessageEvent(
        source="user",
        llm_message=Message(role="user", content=[TextContent(text="hi")]),
    )
    breakdown = build_step_breakdown([message_event], metrics)
    assert breakdown == []
