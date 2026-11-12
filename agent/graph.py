
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.router import IntentRouter

from client.mcp_client import MCPclient
from client.execution_controller import ExecutionController

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
        self.execution_controller = ExecutionController()

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

            if not content:
                continue

            context.append(message)

        return context[-MAX_ROUTER_CONTEXT_MESSAGES:]

    async def build(self):

        # --------------------------------------------------
        # 1. Discover MCP tools
        # --------------------------------------------------

        all_tools = await self.mcp_client.get_tools()

        print("\nDiscovered MCP tools:")

        for tool in all_tools:

            domain = self.mcp_client.get_tool_domain(
                tool.name
            )

            print(
                f"- {tool.name} [{domain or 'unclassified'}]"
            )

        # --------------------------------------------------
        # 2. Intent router
        # --------------------------------------------------

        async def route_intent(state: AgentState):

            context = self.get_router_context(
                state["messages"]
            )

            route = await self.intent_router.route(
                context
            )

            print("\nWorkflow:", route["workflow"])
            print("Domains:", route["domains"])

            return {
                "route": route
            }

        # --------------------------------------------------
        # 3. Dynamic LLM reasoning
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

            # Python selects tools available to this request.
            available_tools = (
                self.mcp_client.get_tools_for_domains(
                    domains
                )
            )

            prompt = get_orchestrator_prompt(
                workflow,
                domains,
            )

            # LLM dynamically chooses the next tool.
            if available_tools:

                model = self.model.bind_tools(
                    available_tools,
                    parallel_tool_calls=False,
                )

            else:

                model = self.model

            response = await (
                prompt | model
            ).ainvoke({
                "messages": state["messages"]
            })

            return {
                "messages": [response],
                "reasoning_cycles": (
                    state.get("reasoning_cycles", 0) + 1
                ),
            }

        # --------------------------------------------------
        # 4. Python controls tool execution
        # --------------------------------------------------

        async def execute_tools(state: AgentState):

            last_message = state["messages"][-1]

            route = state.get("route", {})
            domains = route.get("domains", [])

            available_tools = (
                self.mcp_client.get_tools_for_domains(
                    domains
                )
            )

            result = await self.execution_controller.execute(
                ai_message=last_message,
                available_tools=available_tools,
                state=state,
            )

            if result.get("limit_reached"):
                print("\nTool-call limit reached.")

            return result

        # --------------------------------------------------
        # 5. Decide next graph node
        # --------------------------------------------------

        def route_after_executor(state: AgentState):

            last_message = state["messages"][-1]

            tool_calls = getattr(
                last_message,
                "tool_calls",
                [],
            )

            # LLM has finished.
            if not tool_calls:
                return "finish"

            # Reasoning limit.
            if (
                state.get("reasoning_cycles", 0)
                >= MAX_REASONING_CYCLES
            ):
                return "limit"

            # Tool-call limit.
            if (
                state.get("tool_calls_used", 0)
                >= MAX_TOOL_CALLS
            ):
                return "limit"

            return "execute"

        def route_after_execution(state: AgentState):

            if state.get("limit_reached", False):
                return "limit"

            # Always return tool results to the LLM.
            return "executor"

        # --------------------------------------------------
        # 6. Handle execution limits
        # --------------------------------------------------

        async def handle_limit(state: AgentState):

            last_message = state["messages"][-1]

            tool_calls = getattr(
                last_message,
                "tool_calls",
                [],
            )

            blocked_messages = []

            for call in tool_calls:

                blocked_messages.append(
                    ToolMessage(
                        content=(
                            "Execution stopped because a configured "
                            "limit was reached. Do not call more tools. "
                            "Explain what could and could not be completed."
                        ),
                        tool_call_id=call["id"],
                        name=call["name"],
                        status="error",
                    )
                )

            updated_messages = (
                state["messages"] + blocked_messages
            )

            route = state.get("route", {})

            prompt = get_orchestrator_prompt(
                route.get("workflow", "general"),
                route.get("domains", []),
            )

            # Final response without tools.
            response = await (
                prompt | self.model
            ).ainvoke({
                "messages": updated_messages
            })

            return {
                "messages": [
                    *blocked_messages,
                    response,
                ],
                "limit_reached": True,
            }

        # --------------------------------------------------
        # 7. Build graph
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
            "execute",
            execute_tools,
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

        builder.add_edge(
            "limit_handler",
            END,
        )

        return builder.compile()
