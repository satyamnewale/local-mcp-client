from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.router import IntentRouter

from client.mcp_client import MCPclient
import json

from prompt.orchestrator import get_orchestrator_prompt

from config.limits import (
    MAX_TOOL_CALLS,
    MAX_RETRIES,
    MAX_REASONING_CYCLES,
    MAX_ROUTER_CONTEXT_MESSAGES,
)


class MCPAgent:

    def __init__(self):

        self.mcp_client = MCPclient()

        self.model = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
        )

        self.intent_router = IntentRouter(
            ChatOpenAI(
                temperature=0,
                model="gpt-4o-mini",
            )
        )

    @staticmethod
    def get_router_context(messages: list) -> list:
        """
        Keep recent user/assistant messages for intent classification.

        Tool messages are excluded to avoid sending large tool outputs
        to the router.
        """

        context = []

        for message in messages:

            if isinstance(message, dict):

                role = message.get("role")
                content = message.get("content")

            else:

                message_type = getattr(message, "type", None)

                role_map = {
                    "human": "user",
                    "ai": "assistant",
                }

                role = role_map.get(
                    message_type,
                    message_type,
                )

                content = getattr(message, "content", None)

            if role not in ("user", "assistant"):
                continue

            if content is None or content == "":
                continue

            context.append(message)

        return context[-MAX_ROUTER_CONTEXT_MESSAGES:]

    @staticmethod
    def is_tool_failure(message) -> bool:
        """
        Detect both ToolNode exceptions and structured MCP error results.
        """

        if getattr(message, "status", "success") == "error":
            return True

        content = getattr(message, "content", None)
        artifact = getattr(message, "artifact", None)

        payload = artifact

        if isinstance(content, dict):
            payload = content

        elif isinstance(content, str):
            try:
                parsed = json.loads(content)

                if isinstance(parsed, dict):
                    payload = parsed

            except (ValueError, TypeError):
                pass

        if isinstance(payload, dict):

            if payload.get("ok") is False:
                return True

            if payload.get("success") is False:
                return True

            if payload.get("isError") is True:
                return True

        return False

    async def build(self):

        # --------------------------------------------------
        # 1. Discover MCP tools
        # --------------------------------------------------

        all_tools = await self.mcp_client.get_tools()

        print("\nDiscovered MCP tools:")

        for tool in all_tools:
            domain = self.mcp_client.get_tool_domain(tool.name)
            print(f"- {tool.name} [{domain or 'unclassified'}]")

        # --------------------------------------------------
        # 2. Router node
        # --------------------------------------------------

        async def route_intent(state: AgentState):

            context = self.get_router_context(state["messages"])

            route = await self.intent_router.route(context)

            print("\nSelected workflow:", route["workflow"])
            print("Selected domains:", route["domains"])

            return {"route": route}

        # --------------------------------------------------
        # 3. Executor node
        # --------------------------------------------------

        async def call_executor(state: AgentState):

            route = state.get("route", {})

            workflow = route.get(
                "workflow",
                "general",
            )

            domains = route.get(
                "domains",
                [],
            )

            # Restrict the executor to the selected MCP domains.
            selected_tools = self.mcp_client.get_tools_for_domains(domains)

            prompt = get_orchestrator_prompt(
                workflow,
                domains,
            )

            # Bind only the tools selected by the router.
            if selected_tools:

                model_with_tools = self.model.bind_tools(
                    selected_tools,
                    parallel_tool_calls=False,
                )

            else:

                model_with_tools = self.model

            agent_chain = prompt | model_with_tools

            response = await agent_chain.ainvoke(state["messages"])

            return {
                "messages": [response],
                "reasoning_cycles": (state.get("reasoning_cycles", 0) + 1),
            }

        # --------------------------------------------------
        # 4. Decide whether to call tools, finish, or stop
        # --------------------------------------------------

        def route_after_executor(state: AgentState):

            last_message = state["messages"][-1]

            tool_calls = getattr(
                last_message,
                "tool_calls",
                [],
            )

            # No requested tools: finish normally.
            if not tool_calls:
                return "finish"

            # Reasoning cycle limit.
            if state.get("reasoning_cycles", 0) >= MAX_REASONING_CYCLES:
                return "limit"

            # Tool-call limit.
            if state.get("tool_calls_used", 0) + len(tool_calls) > MAX_TOOL_CALLS:
                return "limit"

            # Retry limit.
            failed_names = state.get(
                "failed_tool_names",
                [],
            )

            pending_retries = sum(call["name"] in failed_names for call in tool_calls)

            if state.get("retries_used", 0) + pending_retries > MAX_RETRIES:
                return "limit"

            return "tools"

        # --------------------------------------------------
        # 5. Tool execution node
        # --------------------------------------------------

        async def run_tools(state: AgentState):

            last_message = state["messages"][-1]

            tool_calls = getattr(
                last_message,
                "tool_calls",
                [],
            )

            route = state.get("route", {})
            domains = route.get("domains", [])

            selected_tools = self.mcp_client.get_tools_for_domains(domains)

            if not selected_tools:
                raise RuntimeError(
                    "The executor requested a tool, "
                    "but no tools are available for the selected domains."
                )

            tool_node = ToolNode(
                selected_tools,
                handle_tool_errors=True,
            )

            result = await tool_node.ainvoke({"messages": [last_message]})

            tool_messages = result.get(
                "messages",
                [],
            )

            failed_names = list(state.get("failed_tool_names", []))

            retries_used = state.get(
                "retries_used",
                0,
            )

            # Match tool results with the original tool calls.
            call_names = {call["id"]: call["name"] for call in tool_calls}

            retry_count = sum(call["name"] in failed_names for call in tool_calls)

            # Update failed tool tracking.
            for message in tool_messages:

                tool_name = call_names.get(message.tool_call_id)

                if not tool_name:
                    continue

                if self.is_tool_failure(message):

                    if tool_name not in failed_names:
                        failed_names.append(tool_name)

                else:

                    if tool_name in failed_names:
                        failed_names.remove(tool_name)

            return {
                "messages": tool_messages,
                "tool_calls_used": (state.get("tool_calls_used", 0) + len(tool_calls)),
                "retries_used": (retries_used + retry_count),
                "failed_tool_names": failed_names,
            }

        # --------------------------------------------------
        # 6. Handle exhausted limits
        # --------------------------------------------------

        async def handle_limit(state: AgentState):

            last_message = state["messages"][-1]

            tool_calls = getattr(
                last_message,
                "tool_calls",
                [],
            )

            if state.get("reasoning_cycles", 0) >= MAX_REASONING_CYCLES:

                reason = "Maximum reasoning cycles reached."

            elif state.get("tool_calls_used", 0) + len(tool_calls) > MAX_TOOL_CALLS:

                reason = "Maximum tool-call limit reached."

            else:

                reason = "Maximum retry limit reached."

            print("\nExecution stopped:", reason)

            # Resolve pending tool calls so the conversation history
            # remains valid for the final response.
            blocked_messages = [
                ToolMessage(
                    content=(
                        f"Tool execution was blocked. {reason} "
                        "Do not request more tools. "
                        "Provide the best response using the "
                        "information already available."
                    ),
                    tool_call_id=call["id"],
                    status="error",
                )
                for call in tool_calls
            ]

            updated_messages = state["messages"] + blocked_messages

            route = state.get("route", {})

            prompt = get_orchestrator_prompt(
                route.get("workflow", "general"),
                route.get("domains", []),
            )

            # Generate a final response without binding tools.
            final_response = await (prompt | self.model).ainvoke(
                {"messages": updated_messages}
            )

            return {
                "messages": [
                    *blocked_messages,
                    final_response,
                ],
                "limit_reached": True,
            }

        # --------------------------------------------------
        # 7. Build LangGraph
        # --------------------------------------------------

        builder = StateGraph(AgentState)

        builder.add_node(
            "router",
            route_intent,
        )

        builder.add_node(
            "executor",
            call_executor,
        )

        builder.add_node(
            "tools",
            run_tools,
        )

        builder.add_node(
            "limit_handler",
            handle_limit,
        )

        builder.add_edge(
            START,
            "router",
        )

        builder.add_edge(
            "router",
            "executor",
        )

        builder.add_conditional_edges(
            "executor",
            route_after_executor,
            {
                "tools": "tools",
                "limit": "limit_handler",
                "finish": END,
            },
        )

        builder.add_edge(
            "tools",
            "executor",
        )

        builder.add_edge(
            "limit_handler",
            END,
        )

        return builder.compile()
