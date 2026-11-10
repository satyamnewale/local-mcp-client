from typing import Any
from langgraph.graph import MessagesState


class AgentState(MessagesState):

    route: dict[str, Any]

    tool_calls_used: int
    reasoning_cycles: int
    retries_used: int

    failed_tool_names: list[str]

    limit_reached: bool