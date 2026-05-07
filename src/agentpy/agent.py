import json
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from .tools import Tools


@dataclass
class Agent:
    """
    An AI agent that can interact with a model via an API, use tools, and maintain context.
    """
    system_prompt: str = "You are my software development mentor"
    model: str = "gemma4:e2b"
    base_url: str = "http://localhost:11434"
    api_key: str = field(default="NO_API_KEY", repr=False)
    tools: Tools = field(default_factory=Tools)
    contexts: dict[str, Callable[[], str]] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register a tool with the agent."""
        return self.tools.register(func)

    def context(self, func: Callable[[], str]) -> Callable[[], str]:
        """Decorator to register a context provider with the agent."""
        self.contexts[func.__name__] = func
        return func

    def chat(self, user_message: str) -> typing.Generator[dict[str, Any], None, str]:
        """
        Send a message to the agent and stream its response.
        Yields:
            dict: {"type": "content", "chunk": str} or {"type": "tool_call", "name": str, ...}
        Returns:
            str: The full accumulated message content.
        """
        self.messages.append({"role": "user", "content": user_message})

        active_contexts = []
        for name, func in self.contexts.items():
            try:
                content = func().strip()
                if content:
                    active_contexts.append(f"<{name}>\n{content}\n</{name}>")
            except Exception as e:
                active_contexts.append(f"<{name}>\nError: {e}\n</{name}>")

        full_system_prompt = self.system_prompt
        if active_contexts:
            context_str = "\n\n".join(active_contexts)
            full_system_prompt += (
                f"\n\nRelevant Context:\n{context_str}\n\n"
                "Use the provided context to answer questions about the current state, user, or environment."
            )

        while True:
            messages = [{"role": "system", "content": full_system_prompt}] + self.messages
            api_kwargs = {
                "model": self.model, 
                "messages": messages,
                "think": True  # Enable native thinking support
            }
            tool_schema = self.tools.get_schemas()

            if tool_schema:
                api_kwargs["tools"] = tool_schema

            url = f"{self.base_url}/api/chat"

            r = self._session.post(
                url,
                json={**api_kwargs, "stream": True},
                timeout=300,
                stream=True,
            )
            r.raise_for_status()

            full_content = ""
            full_thinking = ""
            tool_calls = []

            for line in r.iter_lines():
                if not line:
                    continue
                
                data = json.loads(line)
                message = data.get("message", {})
                
                # Handle official thinking field
                thinking_chunk = message.get("thinking", "")
                if thinking_chunk:
                    full_thinking += thinking_chunk
                    yield {"type": "thinking", "chunk": thinking_chunk}

                content_chunk = message.get("content", "")
                if content_chunk:
                    full_content += content_chunk
                    yield {"type": "content", "chunk": content_chunk}

                # Handle tool calls
                incoming_tool_calls = message.get("tool_calls")
                if incoming_tool_calls:
                    for tc in incoming_tool_calls:
                        tool_calls.append(tc)

                if data.get("done"):
                    break

            # Store in history. Prepend thinking to content wrapped in tags 
            # to ensure the model sees its own reasoning in future turns.
            stored_content = full_content
            if full_thinking:
                stored_content = f"<think>\n{full_thinking}\n</think>\n{full_content}"

            self.messages.append(
                {
                    "role": "assistant",
                    "content": stored_content,
                    "tool_calls": tool_calls,
                }
            )

            if not tool_calls:
                return full_content

            for tool_call in tool_calls:
                yield {"type": "tool_call", "name": tool_call.get("function", {}).get("name")}
                result = self.tools.execute(tool_call)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": json.dumps(result),
                    }
                )
