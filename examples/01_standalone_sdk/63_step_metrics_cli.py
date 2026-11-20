import argparse
import os

from pydantic import SecretStr
from rich.console import Console

from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.conversation.metrics_report import (
    export_metrics_report,
    render_step_metrics_table,
)
from openhands.sdk.conversation.step_metrics import build_step_breakdown
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--export-metrics",
    dest="export_metrics",
    default=None,
    help="Path to write a step-metrics report to on exit (.json for JSON, "
    "anything else for Markdown).",
)
args = parser.parse_args()

# Configure LLM
api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
model = os.getenv("LLM_MODEL", "gpt-5.5")
base_url = os.getenv("LLM_BASE_URL")
llm = LLM(
    usage_id="agent",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
)

cwd = os.getcwd()
tools = [
    Tool(name=TerminalTool.name),
    Tool(name=FileEditorTool.name),
]

agent = Agent(llm=llm, tools=tools)
conversation = Conversation(agent=agent, workspace=cwd)
console = Console()

while True:
    try:
        line = input("> ")
    except EOFError:
        break

    if line == "/exit":
        break

    if line == "/metrics":
        breakdown = build_step_breakdown(conversation.state.events, llm.metrics)
        console.print(render_step_metrics_table(breakdown, llm.metrics))
        continue

    conversation.send_message(line)
    conversation.run()

if args.export_metrics:
    format = "json" if args.export_metrics.endswith(".json") else "markdown"
    breakdown = build_step_breakdown(conversation.state.events, llm.metrics)
    export_metrics_report(breakdown, llm.metrics, args.export_metrics, format=format)
    print(f"Wrote step-metrics report to {args.export_metrics}")

print(f"EXAMPLE_COST: {llm.metrics.accumulated_cost}")
