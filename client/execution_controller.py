import json
import os
import sys
import time

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from config.limits import MAX_TOOL_CALLS


class ExecutionController:
    def __init__(self):
        self.max_tool_calls = MAX_TOOL_CALLS
        self.debug = os.getenv("MCP_DEBUG", "").strip() == "1"

    @staticmethod
    def make_tool_message(
        tool_call: dict,
        content: str,
        status: str = "error",
    ) -> ToolMessage:
        return ToolMessage(
            content=content,
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
            status=status,
        )

    @staticmethod
    def result_status(message: ToolMessage) -> str:
        if getattr(message, "status", "success") == "error":
            return "error"

        content = getattr(message, "content", None)
        if not isinstance(content, str):
            return "returned"

        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            return "returned"

        if isinstance(payload, dict):
            if payload.get("ok") is False or payload.get("success") is False:
                return "reported_failure"

        return "returned"

    async def execute(
        self,
        ai_message,
        available_tools: list,
        state: dict,
    ) -> dict:
        tool_calls = getattr(ai_message, "tool_calls", [])

        if not tool_calls:
            return {"messages": []}

        if len(tool_calls) != 1:
            return {
                "messages": [
                    self.make_tool_message(
                        call,
                        "Only one tool may run at a time. "
                        "Choose one operation, read its result, "
                        "then choose the next operation.",
                    )
                    for call in tool_calls
                ]
            }

        call = tool_calls[0]
        tool_name = call["name"]
        arguments = call.get("args", {})
        calls_used = state.get("tool_calls_used", 0)

        if calls_used >= self.max_tool_calls:
            return {
                "messages": [
                    self.make_tool_message(
                        call,
                        "The tool-call limit was reached. Do not call more tools.",
                    )
                ],
                "limit_reached": True,
            }

        tool_map = {tool.name: tool for tool in available_tools}

        if tool_name not in tool_map:
            return {
                "messages": [
                    self.make_tool_message(
                        call,
                        f"Tool '{tool_name}' is not available.",
                    )
                ]
            }

        if not isinstance(arguments, dict):
            return {
                "messages": [
                    self.make_tool_message(
                        call,
                        "Tool arguments must be an object.",
                    )
                ]
            }

        if self.debug:
            print(
                "[MCP] call "
                f"tool={tool_name} "
                f"args={json.dumps(arguments, ensure_ascii=False, default=str)}",
                file=sys.stderr,
            )

        started = time.perf_counter()

        try:
            result = await ToolNode(
                available_tools,
                handle_tool_errors=True,
            ).ainvoke({
                "messages": [ai_message],
            })

            tool_messages = result.get("messages", [])
            if not tool_messages:
                tool_message = self.make_tool_message(
                    call,
                    "Tool execution returned no result.",
                )
            else:
                tool_message = tool_messages[0]

        except Exception as exc:
            tool_message = self.make_tool_message(
                call,
                f"Tool execution failed: {exc}",
            )

        if self.debug:
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            content_length = len(str(getattr(tool_message, "content", "")))
            print(
                "[MCP] result "
                f"tool={tool_name} "
                f"status={self.result_status(tool_message)} "
                f"chars={content_length} "
                f"elapsed_ms={elapsed_ms}",
                file=sys.stderr,
            )

        return {
            "messages": [tool_message],
            "tool_calls_used": calls_used + 1,
        }