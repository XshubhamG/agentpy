import getpass
import datetime
from dataclasses import dataclass, field
from typing import Callable, Any

import requests
from rich.console import Console


@dataclass
class Agent:
    system_prompt: str = "You are my software development mentor"
    model: str = "gemma4:e2b"
    baseUrl: str = "http://localhost:11434"
    api_key: str = field(default="NO_API_KEY", repr=False)
    contexts: dict[str, Callable[[], str]] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.baseUrl = self.baseUrl.rstrip("/")

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

        messages = [{"role": "system", "content": full_system_prompt}] + self.messages

        url = f"{self.baseUrl}/api/chat"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        r = requests.post(
            url,
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
            timeout=300,
        )

        r.raise_for_status()
        data = r.json()

        message = data.get("message")
        if message is None:
            raise RuntimeError("Model response missing message")

        response = message.get("content") or ""
        self.messages.append({"role": "assistant", "content": response})
        return response


def main() -> None:
    agent = Agent(
        model="gemma4:e2b", system_prompt="End every message with a random fun fact"
    )

    @agent.context
    def user_context() -> str:
        return (
            f"Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Current user: {getpass.getuser()}\n"
        )

    console = Console()

    with console.status("[dim]Thinking...[/dim]", spinner="arc"):
        response = agent.chat("What time is it and who am I?")

    console.print(f"[blue]Assistant: [/blue] {response}")

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
