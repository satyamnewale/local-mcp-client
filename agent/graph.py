import os
import sys

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent.state import AgentState
from client.execution_controller import ExecutionController
from client.mcp_client import MCPclient
from config.limits import MAX_REASONING_CYCLES, MAX_TOOL_CALLS
from prompt.orchestrator import (
    get_orchestrator_prompt,
    get_request_date_context,
)


class MCPAgent:
    def __init__(self):
        self.mcp_client = MCPclient()
        self.execution_controller = ExecutionController()
        self.model = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
        )

    async def build(self):
        all_tools = await self.mcp_client.get_tools()

        if not all_tools:
            raise RuntimeError(
                "No MCP tools were discovered. Check the configured servers."
            )

        print("\nDiscovered MCP tools:")
        for tool in all_tools:
            print(f"- {tool.name}")

        model_with_tools = self.model.bind_tools(
            all_tools,
            parallel_tool_calls=False,
        )

        async def call_executor(state: AgentState):
            date_context = (
                state.get("request_date_context")
                or get_request_date_context()
            )
            prompt = get_orchestrator_prompt(date_context)

            response = await (prompt | model_with_tools).ainvoke({
                "messages": state["messages"],
            })

            if os.getenv("MCP_DEBUG", "").strip() == "1":
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    print(
                        f"[MCP] model_usage={usage}",
                        file=sys.stderr,
                    )

            return {
                "messages": [response],
                "request_date_context": date_context,
                "reasoning_cycles": state.get("reasoning_cycles", 0) + 1,
            }

        async def execute_tools(state: AgentState):
            last_message = state["messages"][-1]
            return await self.execution_controller.execute(
                ai_message=last_message,
                available_tools=all_tools,
                state=state,
            )

        def route_after_executor(state: AgentState):
            last_message = state["messages"][-1]
            tool_calls = getattr(last_message, "tool_calls", [])

            if not tool_calls:
                return "finish"

            if state.get("reasoning_cycles", 0) >= MAX_REASONING_CYCLES:
                return "limit"

            if state.get("tool_calls_used", 0) >= MAX_TOOL_CALLS:
                return "limit"

            return "execute"

        def route_after_execution(state: AgentState):
            if state.get("limit_reached", False):
                return "limit"

            if state.get("tool_calls_used", 0) >= MAX_TOOL_CALLS:
                return "limit"

            return "executor"

        async def handle_limit(state: AgentState):
            last_message = state["messages"][-1]
            pending_calls = getattr(last_message, "tool_calls", [])
            blocked_messages = []

            for call in pending_calls:
                blocked_messages.append(
                    ToolMessage(
                        content=(
                            "This tool call was not run because the configured "
                            "execution limit was reached."
                        ),
                        tool_call_id=call["id"],
                        name=call["name"],
                        status="error",
                    )
                )

            final_message = AIMessage(
                content=(
                    "I reached the execution limit before finishing the request. "
                    "Any action reported as successful above is complete; "
                    "the remaining work may be incomplete."
                )
            )

            return {
                "messages": [*blocked_messages, final_message],
                "limit_reached": True,
            }

        builder = StateGraph(AgentState)

        builder.add_node("executor", call_executor)
        builder.add_node("execute", execute_tools)
        builder.add_node("limit_handler", handle_limit)

        builder.add_edge(START, "executor")

        builder.add_conditional_edges(
            "executor",
            route_after_executor,
            {
                "execute": "execute",
                "limit": "limit_handler",
                "finish": END,
            },
        )

        builder.add_conditional_edges(
            "execute",
            route_after_execution,
            {
                "executor": "executor",
                "limit": "limit_handler",
            },
        )

        builder.add_edge("limit_handler", END)

        return builder.compile()