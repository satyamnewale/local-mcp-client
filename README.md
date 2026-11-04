# Local MCP Client | LangChain + LangGraph

A Python-based local MCP client that connects to multiple locally cloned Model Context Protocol (MCP) servers and makes their tools available to an AI agent using **LangChain and LangGraph**.

The client uses `langchain-mcp-adapters` to connect to MCP servers, discover their tools, and integrate them with an OpenAI model. LangGraph orchestrates the agent workflow, including tool selection, execution, and response generation.

The project currently integrates two independently developed MCP servers:

- [Email MCP](https://github.com/satyamnewale/email-mcp) — Gmail search, email retrieval, conversation retrieval, and attachment downloads.
- [Google Calendar MCP](https://github.com/satyamnewale/google-calendar-mcp) — Calendar event management, event search, recurring events, availability checking, and calendar summaries.

Instead of implementing Gmail and Google Calendar integrations again, this client reuses the existing MCP servers through their local STDIO interfaces.

---

## ✨ Features

- **Multiple MCP Server Connections** — Connect to multiple MCP servers through a single client configuration.
- **Local STDIO Communication** — Launch locally cloned MCP servers as subprocesses and communicate through standard input and output.
- **MCP Tool Discovery** — Automatically discover tools exposed by connected MCP servers.
- **LangChain Integration** — Convert discovered MCP tools into LangChain-compatible tools.
- **LangGraph Agent** — Build an agent workflow using graph-based orchestration.
- **LLM Tool Calling** — Allow the model to select and invoke appropriate tools based on user requests.
- **Conditional Routing** — Route between agent execution, tool execution, and final responses.
- **Multi-Tool Workflow** — Allow the agent to call multiple tools when a request requires them.
- **Conversation History** — Maintain conversation messages during an interactive session.
- **Centralized Configuration** — Manage MCP server commands, arguments, and connection settings in one configuration module.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| Model Context Protocol (MCP) | Communication protocol between client and servers |
| FastMCP | Used by the connected MCP server implementations |
| LangChain | LLM integration and tool binding |
| LangGraph | Agent orchestration and execution workflow |
| LangChain MCP Adapters | MCP connections and tool discovery |
| ChatOpenAI | OpenAI chat model integration |
| asyncio | Asynchronous client execution |
| uv | Python dependency and environment management |
| python-dotenv | Environment variable management |

---

## 🧠 Architecture

```text
                         USER
                           |
                           | Natural Language Request
                           v
                  +--------------------+
                  |      main.py       |
                  |  Client Entry Point|
                  +--------------------+
                           |
                           v
                  +--------------------+
                  |     MCP Agent      |
                  |     graph.py       |
                  +--------------------+
                           |
                           v
                  +--------------------+
                  |     ChatOpenAI     |
                  |  Model + Tool Calls|
                  +--------------------+
                           |
                           v
                  +--------------------+
                  |     LangGraph      |
                  |  Agent Orchestration|
                  +--------------------+
                           |
                   Conditional Routing
                           |
               +-----------+-----------+
               |                       |
               v                       v
          Final Response           ToolNode
                                       |
                                       v
                             LangChain MCP Tools
                                       |
                                       v
                              MCP Client Adapter
                                       |
                         Local STDIO Connections
                                       |
                    +------------------+------------------+
                    |                                     |
                    v                                     v
           +--------------------+               +--------------------+
           |    Email MCP       |               |  Calendar MCP      |
           |    server.py       |               |   server.py        |
           +--------------------+               +--------------------+
                    |                                     |
                    v                                     v
              Gmail API                            Calendar API
                    |                                     |
                    v                                     v
              Gmail Account                       Google Calendar
```

### Architecture Responsibilities

| Component | Responsibility |
|---|---|
| `main.py` | Application entry point and interactive terminal |
| `MCPClient` | Connect to configured MCP servers and discover their tools |
| `MCPAgent` | Initialize the LLM and construct the agent workflow |
| ChatOpenAI | Interpret user requests and generate tool calls |
| LangChain | Bind MCP tools to the LLM |
| LangGraph | Manage state, conditional routing, and tool execution |
| ToolNode | Execute tools selected by the model |
| Email MCP | Handle Gmail operations |
| Calendar MCP | Handle Google Calendar operations |

The client does not directly communicate with the Gmail or Calendar APIs. Those responsibilities remain within their respective MCP server projects.

---

## 🔗 Integrated MCP Servers

### 1. Email MCP

Repository: [satyamnewale/email-mcp](https://github.com/satyamnewale/email-mcp)

A Gmail read-only MCP server built using Python, FastMCP, and Google Gmail API.

**Available functionality:**

| Tool | Description |
|---|---|
| `search_emails` | Search messages by sender, recipient, subject, keywords, dates, time, and other supported conditions |
| `get_email_details` | Retrieve a specific email and its contents |
| `get_email_conversation` | Retrieve messages belonging to an email conversation |
| `get_attachment` | Download an email attachment |

The Email MCP uses the Gmail read-only OAuth scope. It does not expose tools for sending, deleting, or modifying emails.

### 2. Google Calendar MCP

Repository: [satyamnewale/google-calendar-mcp](https://github.com/satyamnewale/google-calendar-mcp)

A Google Calendar MCP server built using Python, FastMCP, and Google Calendar API.

**Available functionality:**

| Tool | Description |
|---|---|
| `create_calendar_event` | Create timed, all-day, multi-day, and recurring events |
| `edit_calendar_event` | Update existing event details |
| `remove_calendar_event` | Delete a calendar event |
| `get_calendar_event` | Retrieve individual event details |
| `postpone_calendar_event` | Postpone an existing event |
| `find_calendar_events` | Search events using supported conditions |
| `find_similar_calendar_events` | Find events with similar titles |
| `get_calendar_overview` | Retrieve calendar summaries |
| `check_calendar_availability` | Check calendar availability and conflicts |
| `suggest_available_slots` | Suggest available time slots |
| `get_calendar_list` | Retrieve available calendars |

For the complete implementation details and tool parameters, refer to the respective MCP repositories.

---

## 📁 Project Structure

```text
mcp-client/
│
├── agent/
│   ├── __init__.py
│   ├── graph.py
│   └── state.py
│
├── client/
│   ├── __init__.py
│   └── mcp_client.py
│
├── config/
│   ├── __init__.py
│   └── mcp_config.py
│
├── main.py
│
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

### Folder Responsibilities

| File / Folder | Responsibility |
|---|---|
| `agent/` | Agent construction and graph management |
| `agent/graph.py` | ChatOpenAI initialization, MCP tool binding, graph nodes, conditional edges, and graph compilation |
| `agent/state.py` | LangGraph conversation state definition |
| `client/` | MCP client connection management |
| `client/mcp_client.py` | Initialize MCP connections and retrieve available tools |
| `config/` | MCP server configuration |
| `config/mcp_config.py` | Define local server commands, arguments, and optional remote connection settings |
| `main.py` | Load environment variables, initialize the agent, and run the interactive terminal |
| `.env` | Store API keys and environment-specific configuration |
| `.env.example` | Example environment variable definitions |
| `pyproject.toml` | Project dependencies and metadata |
| `uv.lock` | Locked dependency versions |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- [Git](https://git-scm.com/)
- OpenAI API key
- Google Cloud OAuth configuration for Email MCP and Calendar MCP
- Both MCP server repositories cloned locally

### 1. Clone the Client Repository

```bash
git clone https://github.com/YOUR_USERNAME/mcp-client.git

cd mcp-client
```

Replace `YOUR_USERNAME` with the GitHub username hosting this client.

### 2. Clone the MCP Server Repositories

Clone the two MCP server projects separately:

```bash
git clone https://github.com/satyamnewale/email-mcp.git
```

```bash
git clone https://github.com/satyamnewale/google-calendar-mcp.git
```

The resulting development environment can be organized as follows:

```text
development/
│
├── mcp-client/
│   ├── agent/
│   ├── client/
│   ├── config/
│   ├── main.py
│   └── ...
│
├── email-mcp/
│   ├── server.py
│   ├── services/
│   └── ...
│
└── google-calendar-mcp/
    ├── server.py
    ├── auth/
    ├── config/
    ├── services/
    └── ...
```

The directory names and locations are examples. You can clone the projects wherever you prefer, provided the paths in the MCP configuration point to the correct locations.

### 3. Set Up the Client Environment

From the client project directory:

```powershell
uv venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Client Dependencies

```powershell
uv sync
```

Alternatively, install the main dependencies manually:

```powershell
uv add langchain langgraph langchain-mcp-adapters langchain-openai python-dotenv
```

---

## ⚙️ MCP Server Configuration

The client uses a centralized configuration module to define which MCP servers should be connected.

Configuration is maintained in `config/mcp_config.py`.

### Example: Local STDIO Configuration

```python
MCP_SERVERS = {

    "email_mcp": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--directory",
            "PATH_TO_EMAIL_MCP",
            "python",
            "server.py"
        ]
    },

    "calendar_mcp": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--directory",
            "PATH_TO_CALENDAR_MCP",
            "python",
            "server.py"
        ]
    }

}
```

Replace the example paths with the actual local directories containing the respective MCP projects.

**Important:** These commands assume that both MCP repositories expose their servers through `python server.py` and have their own dependencies installed through `uv`. If a server requires a different startup command, update its configuration accordingly.

### Configuration Explanation

| Field | Description |
|---|---|
| `email_mcp` | Unique identifier for the Email MCP server |
| `calendar_mcp` | Unique identifier for the Calendar MCP server |
| `transport` | Specifies the MCP transport mechanism |
| `command` | Executable used to launch the server |
| `args` | Arguments passed to the executable |

### Using an Absolute uv Executable

On Windows, you can use the absolute path to the `uv.exe` executable:

```python
"command": r"C:\path\to\uv.exe"
```

This is useful when the terminal environment does not automatically resolve the `uv` command.

### Adding More Local MCP Servers

Additional local MCP servers can be registered by adding another configuration entry.

For example:

```python
"another_mcp": {
    "transport": "stdio",
    "command": "uv",
    "args": [
        "run",
        "--directory",
        "PATH_TO_ANOTHER_MCP",
        "python",
        "server.py"
    ]
}
```

The client can then discover tools exposed by the additional server through the same MCP adapter.

### Remote MCP Configuration

The configuration can also include a remote server using Streamable HTTP.

Example:

```python
import os

MCP_SERVERS = {

    "remote_mcp": {
        "transport": "http",
        "url": os.getenv("REMOTE_MCP_URL"),
        "headers": {
            "Authorization": (
                f"Bearer {os.getenv('REMOTE_MCP_TOKEN')}"
            )
        }
    }

}
```

Configure the remote endpoint and access token in the environment.

The remote configuration is an example of an extension point. The current integration primarily uses locally cloned MCP servers through STDIO.

---

## 🔐 Environment Configuration

Create a `.env` file in the client project root.

```env
OPENAI_API_KEY=your_openai_api_key
```

Example `.env.example`:

```env
OPENAI_API_KEY=

# Optional remote MCP configuration
REMOTE_MCP_URL=
REMOTE_MCP_TOKEN=
```

The OpenAI API key is used by `ChatOpenAI` to communicate with the configured model.

Do not commit `.env` or expose API credentials in source code.

### Google OAuth Credentials

The Email MCP and Calendar MCP repositories manage their own Google authentication.

Each server uses its respective Google OAuth setup.

Refer to their documentation for the required configuration:

- [Email MCP — Google Cloud and OAuth Setup](https://github.com/satyamnewale/email-mcp#google-cloud-setup)
- [Calendar MCP — Google Cloud and OAuth Setup](https://github.com/satyamnewale/google-calendar-mcp#google-cloud-setup)

The client does not implement Google OAuth authentication itself.

For the local deployment, each MCP server accesses its own credentials and token files according to its implementation.

**Important:** The cloned server repositories must be configured and authorized before their Google API tools can operate successfully.

---

## 🔄 Agent Workflow

The client uses LangGraph to orchestrate the tool-calling workflow.

### Execution Sequence

1. The user submits a natural-language request through the terminal.
2. The request is passed to the LangGraph agent.
3. ChatOpenAI interprets the request.
4. The model determines whether an MCP tool is required.
5. If a tool is needed, LangGraph routes execution to the `ToolNode`.
6. The selected tool is invoked through the LangChain MCP adapter.
7. The adapter communicates with the corresponding MCP server.
8. The MCP server performs the requested operation.
9. The result is returned to LangGraph.
10. The agent processes the result and generates a response.

### LangGraph Workflow

```text
              START
                 |
                 v
              AGENT
                 |
                 v
          tools_condition
                 |
          +------+------+
          |             |
          v             v
         END          TOOLS
                        |
                        v
                   MCP Server
                        |
                        v
                   Tool Result
                        |
                        v
                      AGENT
                        |
                        v
                       END
```

The `tools_condition` function determines whether the agent should execute tools or finish the response.

The `ToolNode` handles execution of the tools selected by the model.

---

## ▶️ Running the Client

Once the MCP projects have been cloned and configured, and the client dependencies are installed, start the application from its root directory:

```powershell
uv run python main.py
```

The application initializes the MCP client, discovers configured MCP tools, builds the LangGraph workflow, and starts the interactive terminal.

### Example Startup

```text
==================================================
             MCP CLIENT STARTED
==================================================

Found MCP tools:

- search_emails
- get_email_details
- get_email_conversation
- get_attachment
- create_calendar_event
- edit_calendar_event
- remove_calendar_event
- get_calendar_event
- postpone_calendar_event
- find_calendar_events
- find_similar_calendar_events
- get_calendar_overview
- check_calendar_availability
- suggest_available_slots
- get_calendar_list
```

The exact tools displayed depend on the MCP servers configured and the tools exposed by their current versions.

## 🔧 Adding New MCP Servers

The application is designed to support multiple MCP servers through configuration.

To integrate another server:

1. Clone or obtain the MCP server project locally.
2. Install its dependencies and configure its credentials.
3. Identify its correct startup command and transport.
4. Add the server to `mcp_config.py`.
5. Restart the client.
6. Verify that its tools appear during MCP tool discovery.

The existing LangGraph workflow does not need to be rewritten just to add another server, provided its tools can be loaded through the same adapter and have compatible tool names and schemas.

If two servers expose tools with identical names, tool-name collisions should be handled before combining their tools.

---

## 📚 Documentation and Libraries

### Core Frameworks

| Library | Documentation |
|---|---|
| LangChain | https://docs.langchain.com/oss/python/langchain/overview |
| LangGraph | https://docs.langchain.com/oss/python/langgraph/overview |
| LangChain MCP Adapters | https://github.com/langchain-ai/langchain-mcp-adapters |
| LangChain MCP Integration | https://docs.langchain.com/oss/python/mcp |
| ChatOpenAI | https://docs.langchain.com/oss/python/integrations/chat/openai |
| FastMCP | https://gofastmcp.com/ |
| MCP Python SDK | https://py.sdk.modelcontextprotocol.io/ |
| uv | https://docs.astral.sh/uv/ |
| python-dotenv | https://pypi.org/project/python-dotenv/ |

### Integrated Projects

| Project | Repository |
|---|---|
| Email MCP | https://github.com/satyamnewale/email-mcp |
| Google Calendar MCP | https://github.com/satyamnewale/google-calendar-mcp |

---

## 🤝 Credits and Acknowledgement

This project integrates independently developed MCP servers:

- [Email MCP — Gmail Read-Only Integration](https://github.com/satyamnewale/email-mcp)
- [Google Calendar MCP — Calendar Integration](https://github.com/satyamnewale/google-calendar-mcp)

The client application is built using LangChain, LangGraph, and the LangChain MCP Adapters.

**Documentation and implementation were developed with AI assistance and reviewed by the repository owner.**

**Built with Python, LangChain, LangGraph, and Model Context Protocol.**