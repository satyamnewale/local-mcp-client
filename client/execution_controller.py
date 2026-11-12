
import json
import hashlib

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from config.limits import MAX_TOOL_CALLS, MAX_RETRIES


class ExecutionController:

    def __init__(self):

        self.max_tool_calls = MAX_TOOL_CALLS
        self.max_retries = MAX_RETRIES

    @staticmethod
    def get_signature(tool_name: str, arguments: dict) -> str:

        payload = json.dumps(
            {
                "tool": tool_name,
                "arguments": arguments,
            },
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(
            payload.encode()
        ).hexdigest()

    @staticmethod
    def is_failure(message: ToolMessage) -> bool:

        if getattr(message, "status", "success") == "error":
            return True

        content = getattr(message, "content", None)

        if isinstance(content, dict):
            payload = content

        elif isinstance(content, str):
            try:
                payload = json.loads(content)
            except (ValueError, TypeError):
                return False
        else:
            return False

        if isinstance(payload, dict):
            return (
                payload.get("ok") is False
                or payload.get("success") is False
                or payload.get("isError") is True
            )

        return False

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

    async def execute(
        self,
        ai_message,
        available_tools: list,
        state: dict,
    ) -> dict:

        tool_calls = getattr(
            ai_message,
            "tool_calls",
            [],
        )

        if not tool_calls:
            return {"messages": []}

        # Do not allow parallel execution.
        # The LLM must receive one result before selecting
        # its next dependent action.
        if len(tool_calls) > 1:

            messages = [
                self.make_tool_message(
                    call,
                    (
                        "Multiple tool calls were proposed. "
                        "Execute one tool at a time so that "
                        "each result can inform the next action."
                    ),
                )
                for call in tool_calls
            ]

            return {"messages": messages}

        call = tool_calls[0]

        tool_name = call["name"]
        arguments = call.get("args", {})
        tool_call_id = call["id"]

        # --------------------------------------------------
        # 1. Validate discovered tool
        # --------------------------------------------------

        tool_map = {
            tool.name: tool
            for tool in available_tools
        }

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

        # --------------------------------------------------
        # 2. Execution limit
        # --------------------------------------------------

        calls_used = state.get(
            "tool_calls_used",
            0,
        )

        if calls_used >= self.max_tool_calls:

            return {
                "messages": [
                    self.make_tool_message(
                        call,
                        "Maximum tool-call limit reached.",
                    )
                ],
                "limit_reached": True,
            }

        # --------------------------------------------------
        # 3. Identify exact tool invocation
        # --------------------------------------------------

        signature = self.get_signature(
            tool_name,
            arguments,
        )

        retry_counts = dict(
            state.get("retry_counts", {})
        )

        successful_results = dict(
            state.get("successful_results", {})
        )

        # --------------------------------------------------
        # 4. Prevent duplicate successful execution
        # --------------------------------------------------

        if signature in successful_results:

            cached_result = successful_results[signature]

            return {
                "messages": [
                    ToolMessage(
                        content=cached_result,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                ]
            }

        # --------------------------------------------------
        # 5. Retry control
        # --------------------------------------------------

        failures = retry_counts.get(
            signature,
            0,
        )

        if failures >= self.max_retries:

            return {
                "messages": [
                    self.make_tool_message(
                        call,
                        (
                            "Retry limit reached for this exact "
                            "tool invocation. Do not repeat it."
                        ),
                    )
                ]
            }

        # --------------------------------------------------
        # 6. Execute the tool
        # --------------------------------------------------

        tool_node = ToolNode(
            available_tools,
            handle_tool_errors=True,
        )

        result = await tool_node.ainvoke({
            "messages": [ai_message]
        })

        tool_messages = result.get(
            "messages",
            [],
        )

        if not tool_messages:

            return {
                "messages": [
                    self.make_tool_message(
                        call,
                        "Tool execution returned no result.",
                    )
                ]
            }

        tool_result = tool_messages[0]

        # --------------------------------------------------
        # 7. Inspect actual result
        # --------------------------------------------------

        if self.is_failure(tool_result):

            retry_counts[signature] = failures + 1

        else:

            # Cache only successful results.
            successful_results[signature] = (
                tool_result.content
                if isinstance(tool_result.content, str)
                else json.dumps(
                    tool_result.content,
                    default=str,
                )
            )

        return {
            "messages": [tool_result],
            "tool_calls_used": calls_used + 1,
            "retry_counts": retry_counts,
            "successful_results": successful_results,
            "retries_used": sum(retry_counts.values()),
        }
