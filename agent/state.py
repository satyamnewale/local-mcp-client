
from typing import Any
from langgraph.graph import MessagesState


class AgentState(MessagesState):

    route: dict[str, Any]

    tool_calls_used: int
    reasoning_cycles: int
    retries_used: int

    # Number of failures for each exact tool invocation.
    retry_counts: dict[str, int]

    # Prevent repeating successful operations in the same request.
    successful_results: dict[str, str]

    limit_reached: bool
