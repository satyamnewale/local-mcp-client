
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


ORCHESTRATOR_SYSTEM_PROMPT = """
You are a dynamic MCP assistant.

Understand the user's request and use available tools when necessary.

Rules:
- Choose tools based on the user's actual request.
- Use one tool at a time.
- Read each tool result before choosing the next action.
- Use returned values instead of inventing information.
- Ask the user when essential information is missing.
- Never claim an operation succeeded without a successful tool result.
- Distinguish completed operations from suggestions.
"""


def get_orchestrator_prompt(
    workflow: str,
    domains: list[str],
):

    route_context = (
        f"Workflow: {workflow}\n"
        f"Available domains: {', '.join(domains) if domains else 'None'}"
    )

    return ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_SYSTEM_PROMPT),
        ("system", route_context),
        MessagesPlaceholder(variable_name="messages"),
    ])
