
import asyncio

from dotenv import load_dotenv

load_dotenv()

from agent.graph import MCPAgent
from config.limits import GRAPH_RECURSION_LIMIT


async def main():

    # --------------------------------------------------
    # 1. Initialize MCP Agent
    # --------------------------------------------------

    agent = MCPAgent()

    # Build LangGraph and discover MCP tools.
    graph = await agent.build()

    print("\n" + "=" * 50)
    print("           MCP CLIENT STARTED")
    print("=" * 50)

    # Conversation persists across user requests.
    conversation = []

    # --------------------------------------------------
    # 2. Terminal conversation loop
    # --------------------------------------------------

    while True:

        try:

            user_input = input("\nYou: ").strip()

            if user_input.lower() in {"exit", "quit"}:

                print("\nExiting MCP Client...")
                break

            if not user_input:
                continue

            # Preserve history in case this request fails.
            previous_conversation = conversation.copy()

            # Add current user message.
            conversation.append({
                "role": "user",
                "content": user_input,
            })

            # --------------------------------------------------
            # 3. Initialize per-request agent state
            # --------------------------------------------------

            initial_state = {

                "messages": conversation,

                # Router state
                "route": {},

                # Execution counters
                "tool_calls_used": 0,
                "reasoning_cycles": 0,
                "retries_used": 0,

                # Execution controller state
                "retry_counts": {},
                "successful_results": {},

                # Execution status
                "limit_reached": False,
            }

            # --------------------------------------------------
            # 4. Execute LangGraph
            # --------------------------------------------------

            try:

                result = await graph.ainvoke(
                    initial_state,
                    config={
                        "recursion_limit": GRAPH_RECURSION_LIMIT,
                    },
                )

                # Update conversation with the complete
                # user, assistant and tool message history.
                conversation = result["messages"]

                # Get final assistant response.
                response = conversation[-1]

                print("\nAssistant:")
                print(response.content)

                # --------------------------------------------------
                # 5. Display execution status
                # --------------------------------------------------

                if result.get("limit_reached", False):

                    print(
                        "\n[Execution stopped: "
                        "A configured execution limit was reached.]"
                    )

            except Exception as error:

                # Restore previous conversation if execution fails.
                conversation = previous_conversation

                print(
                    f"\nExecution Error: {error}"
                )

        except KeyboardInterrupt:

            print("\n\nExiting MCP Client...")
            break

        except EOFError:

            print("\n\nInput closed. Exiting MCP Client...")
            break


# --------------------------------------------------
# Application entry point
# --------------------------------------------------

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nMCP Client stopped.")
