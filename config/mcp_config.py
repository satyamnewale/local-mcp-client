import os

MCP_SERVERS = {
    "gmail-mcp": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--directory",
            r"devices\desktop\gmail-mcp",
            "python",
            "server.py"
        ],
    },
    "calendar-mcp": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--directory",
            r"Devices\Desktop\calendar-mcp",
            "python",
            "server.py"
        ],
    },
}