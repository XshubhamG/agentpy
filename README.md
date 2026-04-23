# AgentPy

AgentPy is a lightweight, extensible Python framework for building interactive AI agents powered by local LLMs via Ollama. It provides a simple decorator-based interface for dynamic context injection, allowing agents to stay aware of environment state, user metadata, and real-time data.

## Features

- Dynamic Context Injection: Use the `@agent.context` decorator to provide real-time environment data to the model.
- Session Management: Automatic handling of chat history and message sequencing.
- Ollama Integration: Seamless connection to local LLM deployments.
- Rich Console Interface: Built-in support for stylized terminal interactions.
- Error Resilience: Robust handling of context function failures and API timeouts.

## Installation

This project uses `uv` for dependency management. Ensure you have `uv` installed and then synchronize the environment:

```bash
uv sync
```

Alternatively, you can install the dependencies using `pip`:

```bash
pip install requests rich
```

## Prerequisites

- Ollama must be installed and running on your system.
- The target model (default: `gemma4:e2b`) must be pulled:
  ```bash
  ollama pull gemma4:e2b
  ```

## Usage

The following example demonstrates how to initialize an agent and register a dynamic context function.

```python
from agentpy import Agent
import datetime
import getpass

# Initialize the agent
agent = Agent(
    model="gemma4:e2b",
    system_prompt="You are a helpful assistant."
)

# Register a dynamic context
@agent.context
def environment_context():
    return f"Current Time: {datetime.datetime.now()}\nUser: {getpass.getuser()}"

# Start a chat session
response = agent.chat("What time is it?")
print(response)
```

## API Reference

### Agent Class

The `Agent` class is the primary interface for managing interactions.

#### Configuration
- `system_prompt`: The base instruction set for the model.
- `model`: The name of the model hosted on Ollama.
- `baseUrl`: The endpoint for the Ollama API (defaults to `http://localhost:11434`).
- `api_key`: Optional authentication token for proxied setups.

#### Methods
- `@context`: A decorator used to register functions that return string data to be included in the system prompt.
- `chat(user_message: str) -> str`: Sends a message to the model along with all active contexts and returns the response.

## Development

To run the main interactive loop:

```bash
uv run main.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
