from langchain_mcp_adapters.client import MultiServerMCPClient

from config.mcp_config_main import MCP_SERVERS


DATETIME_TOOLS = {
    "get_current_datetime",
    "get_current_date",
    "get_current_time",
    "get_timezone_info",
    "calculate_relative_date",
    "get_previous_n_days",
    "get_next_n_days",
    "get_week_dates",
    "get_nth_weekday",
    "calculate_relative_datetime",
    "get_date_range",
    "parse_natural_datetime",
    "get_month_dates",
    "calculate_date_difference",
    "calculate_time_difference",
}

GMAIL_TOOLS = {
    "search_emails",
    "get_email_details",
    "get_email_conversation",
    "get_attachment",
}

CALENDAR_TOOLS = {
    "create_calendar_event",
    "edit_calendar_event",
    "remove_calendar_event",
    "get_calendar_event",
    "postpone_calendar_event",
    "find_calendar_events",
    "find_similar_calendar_events",
    "get_calendar_overview",
    "check_calendar_availability",
    "suggest_available_slots",
    "get_calendar_list",
}


class MCPclient:

    def __init__(self):

        self.client = MultiServerMCPClient(MCP_SERVERS)
        self._tools = None

    async def get_tools(self):
        """
        Discover MCP tools once and cache them.
        """

        if self._tools is not None:
            return self._tools

        try:
            tools = await self.client.get_tools()

            if not tools:
                raise RuntimeError("No MCP tools were discovered.")

            self._tools = tools

            return self._tools

        except Exception as exc:
            raise RuntimeError(
                f"Unable to discover MCP tools: {exc}"
            ) from exc

    @staticmethod
    def get_tool_domain(tool_name: str) -> str | None:
        """
        Identify the MCP domain using the discovered tool name.

        Supports names with server prefixes, for example:
        - search_emails
        - gmail_search_emails
        - gmail__search_emails
        """

        name = tool_name.lower().replace("-", "_")

        tool_groups = {
            "datetime": DATETIME_TOOLS,
            "gmail": GMAIL_TOOLS,
            "calendar": CALENDAR_TOOLS,
        }

        for domain, known_names in tool_groups.items():

            for known_name in known_names:

                if (
                    name == known_name
                    or name.endswith(f"_{known_name}")
                    or name.endswith(f".{known_name}")
                ):
                    return domain

        return None

    def get_tools_for_domains(
        self,
        domains: list[str],
    ) -> list:

        if self._tools is None:
            raise RuntimeError(
                "MCP tools have not been initialized. "
                "Call get_tools() first."
            )

        selected_domains = set(domains)

        if not selected_domains:
            return []

        selected_tools = [
            tool
            for tool in self._tools
            if self.get_tool_domain(tool.name) in selected_domains
        ]

        found_domains = {
            self.get_tool_domain(tool.name)
            for tool in selected_tools
        }

        missing_domains = selected_domains - found_domains

        if missing_domains:
            raise RuntimeError(
                "No tools found for MCP domain(s): "
                + ", ".join(sorted(missing_domains))
            )

        return selected_tools

    async def get_tools_name(self) -> list[str]:

        tools = await self.get_tools()

        return [
            tool.name
            for tool in tools
        ]