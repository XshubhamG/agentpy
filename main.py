import datetime
import getpass
import inspect
import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Final, Union, get_args, get_origin

import requests
from rich.console import Console


@dataclass
class Tools:
    TOOL_SCHEMA_ATTR: Final[str] = "__tool_schema__"
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)

    @staticmethod
    def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "string"}
        description: str | None = None
        origin = get_origin(annotation)

        if origin is Annotated:
            base_type, *meta = get_args(annotation)
            schema = Tools._annotation_to_schema(base_type)
            if meta:
                description = str(meta[0])
        elif annotation in (int, float):
            schema = {"type": "number"}
        elif annotation is bool:
            schema = {"type": "boolean"}
        elif annotation is str:
            schema = {"type": "string"}
        elif annotation is dict:
            schema = {"type": "object"}
        elif annotation is list:
            schema = {"type": "array"}
        elif origin is list:
            schema = {
                "type": "array",
                "items": Tools._annotation_to_schema(get_args(annotation)[0]),
            }
        elif origin is dict:
            schema = {"type": "object"}

        elif origin is Union:
            any_of = [
                Tools._annotation_to_schema(arg)
                for arg in get_args(annotation)
                if arg is not type(None)
            ]
            if any_of:
                schema = any_of[0]

        if description:
            schema["description"] = description

        return schema

    @classmethod
    def schema_for_callable(cls, func: Callable[..., Any]) -> dict[str, Any]:
        sig = inspect.signature(func)
        annotations = inspect.get_annotations(func)

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        for name, param in sig.parameters.items():
            annotation = annotations.get(name, inspect.Parameter.empty)

            if annotation is inspect.Parameter.empty:
                continue

            parameters["properties"][name] = cls._annotation_to_schema(annotation)

            if param.default is param.empty:
                parameters["required"].append(name)

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "No description provided.",
                "parameters": parameters,
                "strict": True,
            },
        }

    def get_schemas(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for fn in self.tools.values():
            s = getattr(fn, self.TOOL_SCHEMA_ATTR, None)
            if s is not None:
                out.append(s)

        return out

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Attach generated schema to ``func`` (if missing), register by name; returns ``func``."""
        if getattr(func, self.TOOL_SCHEMA_ATTR, None) is None:
            setattr(func, self.TOOL_SCHEMA_ATTR, self.schema_for_callable(func))
        self.tools[func.__name__] = func
        return func

    def execute(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Run a tool from a chat-completions ``tool_calls`` entry (name + arguments JSON)."""
        fn_payload = tool_call.get("function") or {}
        fn_name = fn_payload.get("name")
        fn = self.tools.get(fn_name) if fn_name else None
        if not fn:
            return {"error": f"Tool '{fn_name}' not found"}
        try:
            args = json.loads(fn_payload.get("arguments") or "{}")
            result = fn(**args)
            return result if isinstance(result, dict) else {"result": result}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return {"error": str(e)}


@dataclass
class Agent:
    system_prompt: str = "You are my software development mentor"
    model: str = "gemma4:e2b"
    baseUrl: str = "http://localhost:11434"
    api_key: str = field(default="NO_API_KEY", repr=False)
    tools: Tools = field(default_factory=Tools)
    contexts: dict[str, Callable[[], str]] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.baseUrl = self.baseUrl.rstrip("/")

    def tool(self, func: Callable[[], str]) -> Callable[..., Any]:
        return self.tools.register(func)

    def context(self, func: Callable[[], str]) -> Callable[[], str]:
        self.contexts[func.__name__] = func
        return func

    def chat(self, user_message: str) -> str:
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
            api_kwargs = {"model": self.model, "messages": messages}
            tool_schema = self.tools.get_schemas()

            if tool_schema:
                api_kwargs["tools"] = tool_schema

            url = f"{self.baseUrl}/api/chat"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            r = requests.post(
                url,
                headers=headers,
                json={**api_kwargs, "stream": False},
                timeout=300,
            )

            r.raise_for_status()
            data = r.json()

            message = data.get("message")
            if message is None:
                raise RuntimeError("Model response missing message")

            response = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            self.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "tool_calls": [
                        {
                            "id": tc.get("id"),
                            "type": tc.get("type"),
                            "function": {
                                "name": (tc.get("function") or {}).get("name"),
                                "arguments": (tc.get("function") or {}).get(
                                    "arguments"
                                ),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            if not tool_calls:
                return response

            for tool_call in tool_calls:
                result = self.tools.execute(tool_call)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": json.dumps(result),
                    }
                )


def main() -> None:
    agent = Agent(
        model="gemma4:e2b", system_prompt="Be a very good personal ai assistant"
    )

    @agent.context
    def user_context() -> str:
        return (
            f"Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Current user: {getpass.getuser()}\n"
        )

    @agent.tool
    def add(
        a: Annotated[int, "First number"], b: Annotated[int, "Second number"]
    ) -> dict[str, int]:
        """Add two numbers together"""
        return {"result": a + b}

    @agent.tool
    def multiply(
        a: Annotated[int, "First number"], b: Annotated[int, "Second number"]
    ) -> dict[str, int]:
        """Multiply two numbers together"""
        return {"result": a * b}

    @agent.tool
    def secret() -> dict[str, str]:
        """Return secret key"""
        return {"result": "fluffy bunnies"}

    console = Console()

    while True:
        console.print("[green]You: [/green]", end="")
        user_input = console.input()

        if user_input.strip().lower() in {"quit", "exit"}:
            console.print("[dim]Goodbye![/dim]")
            return

        with console.status("[dim]Thinking...[/dim]", spinner="arc"):
            response = agent.chat(user_input).strip()

        console.print(f"[blue]Assistant: [/blue] {response}")


if __name__ == "__main__":
    main()
