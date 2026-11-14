from langchain_mcp_adapters.client import MultiServerMCPClient

from config.mcp_config_main import MCP_SERVERS


class MCPclient:
    def __init__(self):
        self.client = MultiServerMCPClient(
            MCP_SERVERS,
            tool_name_prefix=True,
        )
        self._tools = None
        self._tools_by_server = {}

    async def get_tools(self) -> list:
        """Discover and cache tools from every configured MCP server."""
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

    async def get_tools_for_domains(self, domains: list[str]) -> list:
        """
        Discover and cache tools from the requested MCP servers.

        Domain names are the keys in MCP_SERVERS, such as "calendar"
        or "datetime". Every tool exposed by those servers is included,
        including tools added after this client code was written.
        """
        selected_domains = list(dict.fromkeys(domains))

        if not selected_domains:
            return []

        unknown_domains = set(selected_domains) - set(MCP_SERVERS)
        if unknown_domains:
            raise ValueError(
                "Unknown MCP server(s): "
                + ", ".join(sorted(unknown_domains))
                + ". Configured servers: "
                + ", ".join(sorted(MCP_SERVERS))
            )

        selected_tools = []

        for domain in selected_domains:
            if domain not in self._tools_by_server:
                try:
                    tools = await self.client.get_tools(server_name=domain)

                    if not tools:
                        raise RuntimeError(
                            f"No tools discovered for MCP server '{domain}'."
                        )

                    self._tools_by_server[domain] = tools

                except Exception as exc:
                    raise RuntimeError(
                        f"Unable to discover tools for MCP server "
                        f"'{domain}': {exc}"
                    ) from exc

            selected_tools.extend(self._tools_by_server[domain])

        return selected_tools

    def get_available_domains(self) -> list[str]:
        """Return the configured MCP server names."""
        return list(MCP_SERVERS)

    async def get_tools_name(self) -> list[str]:
        """Return the names of all discovered tools."""
        tools = await self.get_tools()
        return [tool.name for tool in tools]