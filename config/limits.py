MAX_TOOL_CALLS = 25

MAX_RETRIES = 2

MAX_REASONING_CYCLES = 8

# Router context size
MAX_ROUTER_CONTEXT_MESSAGES = 8

# One router call, executor/tool cycles, and limit handling.
GRAPH_RECURSION_LIMIT = (
    2 * MAX_REASONING_CYCLES + 5
)