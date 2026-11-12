
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_openai import ChatOpenAI


# --------------------------------------------------
# ROUTER SCHEMAS
# --------------------------------------------------

Domain = Literal[
    "datetime",
    "gmail",
    "calendar",
]


Workflow = Literal[
    "datetime",
    "email_search",
    "email_read",
    "attachment_download",
    "email_to_calendar",
    "calendar_search",
    "calendar_availability",
    "calendar_create",
    "calendar_update",
    "calendar_delete",
    "general",
]


class RouteDecision(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    workflow: Workflow = Field(
        description="Primary workflow requested by the user."
    )

    domains: list[Domain] = Field(
        description="MCP domains relevant to the request."
    )

    reason: str = Field(
        description="Short reason for the routing decision."
    )


# --------------------------------------------------
# REQUIRED DOMAIN MAPPING
# --------------------------------------------------

REQUIRED_DOMAINS = {

    "datetime": {
        "datetime",
    },

    "email_search": {
        "gmail",
    },

    "email_read": {
        "gmail",
    },

    "attachment_download": {
        "gmail",
    },

    "email_to_calendar": {
        "datetime",
        "gmail",
        "calendar",
    },

    "calendar_search": {
        "calendar",
    },

    "calendar_availability": {
        "calendar",
    },

    "calendar_create": {
        "calendar",
    },

    "calendar_update": {
        "calendar",
    },

    "calendar_delete": {
        "calendar",
    },

    "general": set(),
}


# --------------------------------------------------
# ROUTER PROMPT
# --------------------------------------------------

ROUTER_SYSTEM_PROMPT = """
You are an intent router for a multi-MCP assistant.

Available domains:

1. datetime
   Date, time, timezone and date calculations.

2. gmail
   Email search, retrieval, conversations and attachments.

3. calendar
   Event search, availability, creation, modification and deletion.

Identify the user's primary workflow and relevant domains.

Available workflows:

- datetime
- email_search
- email_read
- attachment_download
- email_to_calendar
- calendar_search
- calendar_availability
- calendar_create
- calendar_update
- calendar_delete
- general

Rules:

- Consider conversation context, not only the latest message.
- Understand short replies using the previous conversation.
- Include every domain that may be required.
- Relative dates and times may require datetime.
- Email-to-calendar tasks require gmail, datetime and calendar.
- Do not select unrelated domains.
- Do not plan tool calls or execution sequences.
- Do not answer the user.

Return only the structured routing decision.
"""


# --------------------------------------------------
# INTENT ROUTER
# --------------------------------------------------

class IntentRouter:

    def __init__(self, model: ChatOpenAI):

        self.model = model

        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            MessagesPlaceholder(
                variable_name="messages"
            ),
        ])

        self.chain = (
            prompt
            | model.with_structured_output(
                RouteDecision
            )
        )

    async def route(
        self,
        messages: list,
    ) -> dict:

        # ------------------------------------------
        # 1. LLM identifies intent and domains
        # ------------------------------------------

        decision: RouteDecision = (
            await self.chain.ainvoke({
                "messages": messages
            })
        )

        workflow = decision.workflow

        selected_domains = set(
            decision.domains
        )

        # ------------------------------------------
        # 2. Enforce mandatory domains
        # ------------------------------------------

        required_domains = REQUIRED_DOMAINS.get(
            workflow,
            set(),
        )

        selected_domains.update(
            required_domains
        )

        # ------------------------------------------
        # 3. Return normalized route
        # ------------------------------------------

        route = {
            "workflow": workflow,

            "domains": sorted(
                selected_domains
            ),

            "reason": decision.reason,
        }

        return route
