
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from prompt.orchestrator import get_orchestrator_prompt

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from client.mcp_client import MCPclient

from config.limits import (
    MAX_TOOL_CALLS,
    MAX_RETRIES,
    MAX_REASONING_CYCLES
)


class MCPAgent:

    def __init__(self):

        self.mcp_client = MCPclient()

        self.model = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini"
        )

    async def build(self):

        # 1. Discover MCP tools
        tools = await self.mcp_client.get_tools()

        if not tools:
            raise RuntimeError("No MCP tools were discovered.")

        print("\nFound MCP tools:")

        for tool in tools:
            print(f"- {tool.name}")

        # 2. Bind tools to the model
        model_with_tools = self.model.bind_tools(tools)

        # Create orchestration prompt
        prompt = get_orchestrator_prompt()

        # Agent chain
        agent_chain = prompt | model_with_tools

        # 3. Initialize ToolNode
        tool_node = ToolNode(tools)

        # 4. Agent node
        async def call_model(state: AgentState):

            response = await agent_chain.ainvoke(
                state["messages"]
            )

            return {
                "messages": [response],
                "reasoning_cycles": (
                    state.get("reasoning_cycles", 0) + 1
                )
            }

        # 5. Check execution limits
        def route_after_agent(state: AgentState):

            last_message = state["messages"][-1]

            tool_calls = getattr(
                last_message,
                "tool_calls",
                []
            )

            # No tool calls: finish normally
            if not tool_calls:
                return "finish"

            requested_calls = len(tool_calls)

            # Reasoning cycle limit
            if state.get("reasoning_cycles", 0) >= MAX_REASONING_CYCLES:
                return "limit"

            # Tool-call limit
            if (
                state.get("tool_calls_used", 0)
                + requested_calls
                > MAX_TOOL_CALLS
            ):
                return "limit"

            # Retry limit
            failed_names = state.get(
                "failed_tool_names", []
            )

            pending_retries = sum(
                call["name"] in failed_names
                for call in tool_calls
            )

            if (
                state.get("retries_used", 0)
                + pending_retries
                > MAX_RETRIES
            ):
                return "limit"

            return "tools"

        # 6. Controlled tool execution
        async def run_tools(state: AgentState):

            last_message = state["messages"][-1]
            tool_calls = last_message.tool_calls

            failed_names = list(
                state.get("failed_tool_names", [])
            )

            retries_used = state.get("retries_used", 0)

            # Count calls that repeat previously failed tools
            retry_count = sum(
                call["name"] in failed_names
                for call in tool_calls
            )

            # Execute tools
            result = await tool_node.ainvoke(state)

            tool_messages = result["messages"]

            # Map tool-call IDs to tool names
            call_names = {
                call["id"]: call["name"]
                for call in tool_calls
            }

            # Track tool failures
            for message in tool_messages:

                tool_name = call_names.get(
                    message.tool_call_id
                )

                if not tool_name:
                    continue

                if getattr(message, "status", "success") == "error":

                    if tool_name not in failed_names:
                        failed_names.append(tool_name)

                else:

                    if tool_name in failed_names:
                        failed_names.remove(tool_name)

            return {
                "messages": tool_messages,

                "tool_calls_used": (
                    state.get("tool_calls_used", 0)
                    + len(tool_calls)
                ),

                "retries_used": (
                    retries_used + retry_count
                ),

                "failed_tool_names": failed_names
            }

        # 7. Handle exhausted execution limits
        async def handle_limit(state: AgentState):

            last_message = state["messages"][-1]

            tool_calls = getattr(
                last_message,
                "tool_calls",
                []
            )

            # Determine why execution stopped
            if state.get("reasoning_cycles", 0) >= MAX_REASONING_CYCLES:
                reason = "Maximum reasoning cycles reached."

            elif (
                state.get("tool_calls_used", 0)
                + len(tool_calls)
                > MAX_TOOL_CALLS
            ):
                reason = "Maximum tool-call limit reached."

            else:
                reason = "Maximum retry limit reached."

            print(f"\nExecution stopped: {reason}")

            # Resolve pending tool calls so the message
            # history remains valid for the next LLM call.
            blocked_messages = [
                ToolMessage(
                    content=(
                        f"Tool execution was blocked. {reason} "
                        "Do not request more tools. "
                        "Provide the best response using "
                        "the information already available."
                    ),
                    tool_call_id=call["id"]
                )
                for call in tool_calls
            ]

            updated_messages = (
                state["messages"] + blocked_messages
            )

            # Generate final response WITHOUT tools
            final_response = await (prompt | self.model).ainvoke(
                updated_messages
            )

            return {
                "messages": [
                    *blocked_messages,
                    final_response
                ],
                "limit_reached": True
            }

        # 8. Build LangGraph
        builder = StateGraph(AgentState)

        builder.add_node("agent", call_model)
        builder.add_node("tools", run_tools)
        builder.add_node("limit_handler", handle_limit)

        # Starting point
        builder.add_edge(START, "agent")

        # Conditional routing
        builder.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "tools": "tools",
                "limit": "limit_handler",
                "finish": END
            }
        )

        # Tool results return to agent
        builder.add_edge("tools", "agent")

        # Limit handler returns final answer
        builder.add_edge("limit_handler", END)

        # Compile graph
        graph = builder.compile()

        return graph
