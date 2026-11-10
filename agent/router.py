from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


Domain = Literal["datetime", "gmail", "calendar"]

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
    model_config = ConfigDict(extra="forbid")

    workflow: Workflow = Field(
        description="The primary workflow requested by the user."
    )

    domains: list[Domain] = Field(
        description=(
            "MCP domains needed to complete the request. "
            "Use only datetime, gmail, and calendar."
        )
    )

    reason: str = Field(
        description="Brief explanation for the selected workflow and domains."
    )


ROUTER_SYSTEM_PROMPT = """
You are an intent router for a personal productivity MCP client.

Available MCP domains:

1. datetime
   - Current date and time
   - Date parsing and relative date calculations
   - Time calculations and timezone information

2. gmail
   - Search emails
   - Retrieve email details and conversations
   - Download attachments

3. calendar
   - Search and retrieve calendar events
   - Check availability and suggest available slots
   - Create, edit, postpone, and delete events

Your job is ONLY to classify the request.
Do not answer the user.
Do not execute tools.
Do not generate a workflow or tool-call sequence.

WORKFLOW CLASSIFICATION

- datetime:
  Date, time, relative-date, weekday, or timezone calculations.

- email_search:
  Search or list emails.

- email_read:
  Retrieve the content or details of an email or conversation.

- attachment_download:
  Download an email attachment.

- email_to_calendar:
  Extract meeting/event information from email and create or schedule
  a corresponding calendar event.

- calendar_search:
  Search calendar events or retrieve event details.

- calendar_availability:
  Check whether a requested calendar time is free or find alternatives.

- calendar_create:
  Create a calendar event.

- calendar_update:
  Edit or postpone an existing calendar event.

- calendar_delete:
  Delete a calendar event or recurring occurrence.

- general:
  A normal conversational or informational request that does not need MCP tools.

DOMAIN SELECTION

- Date calculations require datetime.
- Email tasks require gmail.
- Calendar operations require calendar.
- Include datetime when relative dates or times must be resolved before
  passing them to Gmail or Calendar.
- An email-to-calendar workflow requires all three domains:
  datetime, gmail, and calendar.
- Do not select unrelated domains.

CONVERSATION CONTEXT

- Consider recent messages, not only the latest user message.
- A short reply such as "yes", "confirm", or "do it" may refer to an
  action proposed by the assistant in the previous messages.
- Preserve the domain of the pending action when interpreting confirmations.
- A deletion confirmation should be routed to calendar.
- Never interpret a confirmation as permission for a different action.

Return only the structured RouteDecision.
"""


class IntentRouter:

    def __init__(self, model: ChatOpenAI):

        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.chain = prompt | model.with_structured_output(RouteDecision)

    async def route(self, messages: list) -> dict:

        decision: RouteDecision = await self.chain.ainvoke({
            "messages": messages
        })

        domains = set(decision.domains)

        # Enforce the domains required by the workflow.
        required_domains = {
            "datetime": {"datetime"},
            "email_search": {"gmail"},
            "email_read": {"gmail"},
            "attachment_download": {"gmail"},
            "email_to_calendar": {
                "datetime",
                "gmail",
                "calendar",
            },
            "calendar_search": {"calendar"},
            "calendar_availability": {"calendar"},
            "calendar_create": {"calendar"},
            "calendar_update": {"calendar"},
            "calendar_delete": {"calendar"},
            "general": set(),
        }

        domains.update(required_domains[decision.workflow])

        return {
            "workflow": decision.workflow,
            "domains": sorted(domains),
            "reason": decision.reason,
        }