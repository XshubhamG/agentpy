from dataclasses import dataclass, field
from typing import Any

import requests
from rich.console import Console


@dataclass
class Agent:
    model: str = "qwen3.5"
    baseUrl: str = "http://localhost:11434"
    api_key: str = field(default="NO_API_KEY", repr=False)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.baseUrl = self.baseUrl.rstrip("/")

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        url = f"{self.baseUrl}/api/chat"

        r = requests.post(
            url,
            json={
                "model": self.model,
                "messages": self.messages,
                "stream": False,
                "options": {"think": False},
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
    agent = Agent(model="qwen3.5")
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
