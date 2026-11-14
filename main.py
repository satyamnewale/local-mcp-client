import asyncio

from dotenv import load_dotenv

load_dotenv()

from agent.graph import MCPAgent
from config.limits import (
    GRAPH_RECURSION_LIMIT,
    MAX_HISTORY_ASSISTANT_CHARS,
    MAX_HISTORY_MESSAGES,
)
from prompt.orchestrator import get_request_date_context


def compact_conversation(messages: list) -> list[dict[str, str]]:
    """Keep recent user/assistant text, excluding tool payloads and tool calls."""
    history = []

    for message in messages:
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
            tool_calls = message.get("tool_calls", [])
        else:
            role = {
                "human": "user",
                "ai": "assistant",
            }.get(getattr(message, "type", None))
            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", [])

        if role not in {"user", "assistant"} or not content or tool_calls:
            continue

        if not isinstance(content, str):
            content = str(content)

        if role == "assistant" and len(content) > MAX_HISTORY_ASSISTANT_CHARS:
            suffix = "\n[Earlier reply shortened.]"
            content = content[:MAX_HISTORY_ASSISTANT_CHARS - len(suffix)] + suffix

        history.append({
            "role": role,
            "content": content,
        })

    return history[-MAX_HISTORY_MESSAGES:]


async def main():
    agent = MCPAgent()
    graph = await agent.build()

    print("\n" + "=" * 50)
    print("           MCP CLIENT STARTED")
    print("=" * 50)

    conversation = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("\nExiting MCP Client...")
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
                "request_date_context": get_request_date_context(),
                "tool_calls_used": 0,
                "reasoning_cycles": 0,
                "limit_reached": False,
            }

            try:
                result = await graph.ainvoke(
                    initial_state,
                    config={
                        "recursion_limit": GRAPH_RECURSION_LIMIT,
                    },
                )

                conversation = compact_conversation(result["messages"])
                response = result["messages"][-1]

                print("\nAssistant:")
                print(response.content)

                if result.get("limit_reached", False):
                    print("\n[Execution stopped at the configured limit.]")

            except Exception as error:
                conversation = previous_conversation
                print(f"\nExecution Error: {error}")

        except KeyboardInterrupt:
            print("\n\nExiting MCP Client...")
            break
        except EOFError:
            print("\n\nInput closed. Exiting MCP Client...")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMCP Client stopped.")