import asyncio

from dotenv import load_dotenv

load_dotenv()

from agent.graph import MCPAgent

from config.limits import GRAPH_RECURSION_LIMIT


async def main():

    agent = MCPAgent()

    graph = await agent.build()

    print("\n" + "=" * 50)
    print("           MCP CLIENT STARTED")
    print("=" * 50)

    conversation = []

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting MCP Client...")
            break

        if not user_input:
            continue

        previous_conversation = conversation.copy()

        conversation.append({
            "role": "user",
            "content": user_input,
        })

        initial_state = {
            "messages": conversation,

            # Reset per-request execution counters.
            "route": {},
            "tool_calls_used": 0,
            "reasoning_cycles": 0,
            "retries_used": 0,
            "failed_tool_names": [],
            "limit_reached": False,
        }

        try:

            result = await graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": GRAPH_RECURSION_LIMIT
                },
            )

            conversation = result["messages"]

            response = conversation[-1]

            print("\nAssistant:")
            print(response.content)

            if result.get("limit_reached"):
                print("\n[Execution stopped because a configured limit was reached.]")

        except Exception as error:

            # Restore the previous history if this turn failed.
            conversation = previous_conversation

            print(f"\nExecution Error: {error}")


if __name__ == "__main__":
    asyncio.run(main())