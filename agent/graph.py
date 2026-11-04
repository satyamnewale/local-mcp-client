
from langchain_openai import ChatOpenAI

from langgraph.graph import (
    StateGraph,
    START
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from agent.state import AgentState
from client.mcp_client import MCPclient

from dotenv import load_dotenv

load_dotenv()


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

        print("Found MCP tools:")

        for tool in tools:
            print(f"- {tool.name}")

        # 2. Bind MCP tools to LangChain model
        model_with_tools = self.model.bind_tools(tools)

        # 3. Define agent node
        async def call_model(state: AgentState):

            response = await model_with_tools.ainvoke(
                state["messages"]
            )

            return {
                "messages": [response]
            }

        # 4. Initialize LangGraph
        builder = StateGraph(AgentState)

        # 5. Register nodes
        builder.add_node(
            "agent",
            call_model
        )

        builder.add_node(
            "tools",
            ToolNode(tools)
        )

        # 6. Start workflow
        builder.add_edge(
            START,
            "agent"
        )

        # 7. Conditional routing from agent
        builder.add_conditional_edges(
            "agent",
            tools_condition
        )

        # 8. Return tool results to agent
        builder.add_edge(
            "tools",
            "agent"
        )

        # 9. Compile graph
        graph = builder.compile()

        return graph
