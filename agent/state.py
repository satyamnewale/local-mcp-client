from typing import NotRequired

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    request_date_context: NotRequired[str]
    tool_calls_used: int
    reasoning_cycles: int
    limit_reached: NotRequired[bool]