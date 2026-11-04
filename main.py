import asyncio
from dotenv import load_dotenv

load_dotenv()

from agent.graph import MCPAgent


async def main():

    # Initialize agent
    agent = MCPAgent()

    # Build LangGraph workflow and connect MCP tools
    graph = await agent.build()

    print("\n" + "=" * 50)
    print("           MCP CLIENT STARTED")
    print("=" * 50)

    conversation = []

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting MCP Client...")
            break

        if not user_input:
            continue

        # Add user message
        conversation.append({
            "role": "user",
            "content": user_input
        })

        try:

            # Execute LangGraph
            result = await graph.ainvoke({
                "messages": conversation
            })

            # Preserve conversation history
            conversation = result["messages"]

            # Get final assistant response
            response = conversation[-1]

            print("\nAssistant:")
            print(response.content)

        except Exception as error:

            print(f"\nExecution Error: {error}")


if __name__ == "__main__":
    asyncio.run(main())