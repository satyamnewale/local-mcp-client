from langchain_mcp_adapters.client import MultiServerMCPClient
from config.mcp_config_main import MCP_SERVERS

class MCPclient :

    def __init__(self):
        self.client = MultiServerMCPClient(MCP_SERVERS)

    async def get_tools(self) :
        """
        discover tools from all configured MCP servers.
        """

        try :
            tools = await self.client.get_tools()

            return tools
        except Exception as e :
            print(f"Error while getting tools : {e}")

    async def get_tools_name(self):
        """
        discover tools names from all configured MCP servers.
        """

        try :
            tools = await self.client.get_tools()
            return [tool.name for tool in tools]
        except Exception as e :
            print(f"Error while getting tool names : {e}")