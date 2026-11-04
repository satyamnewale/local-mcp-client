import asyncio

from agent.graph import MCPAgent

async def main():

    agent = MCPAgent()

    graph = await agent.build()

    print("\nMCP CLIENT READY")
    print("Type 'exit' to quit.\n")

    conversation = []

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            break

        if not user_input:
            continue

        conversation.append({
            "role": "user",
            "content": user_input
        })

        try:

            response = await graph.ainvoke({
                "messages": conversation
            })

            conversation = response["messages"]

            final_message = conversation[-1]

            print("\nAssistant:")
            print(final_message.content)
            print()

        except Exception as error:

            print(f"\nError: {error}\n")


if __name__ == "__main__":
    asyncio.run(main())