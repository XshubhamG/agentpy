import datetime
import getpass
from typing import Annotated

from rich.console import Console

from .agent import Agent


def main() -> None:
    """CLI entry point for the AgentPy assistant."""
    console = Console()
    agent = Agent(
        model="gemma4:e2b", 
        system_prompt="Be a very good personal AI assistant"
    )

    @agent.context
    def user_context() -> str:
        """Context provider for current time and user."""
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

    console.print("[bold blue]AgentPy CLI[/bold blue]")
    console.print("[dim]Type '/exit' or 'quit' to stop. Type '/clear' to reset chat history.[/dim]\n")

    while True:
        try:
            console.print("[green]You: [/green]", end="")
            user_input = console.input().strip()

            if not user_input:
                continue

            if user_input.lower() in {"/exit", "exit", "quit"}:
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/clear":
                agent.messages = []
                console.print("[yellow]Chat history cleared.[/yellow]")
                continue

            console.print("[blue]Assistant: [/blue]", end="")
            
            from rich.live import Live
            from rich.text import Text

            full_thinking = ""
            full_content = ""

            def render_response(thinking: str, content: str) -> Text:
                """Render thinking (dimmed) and content together."""
                styled_text = Text()
                if thinking:
                    styled_text.append(thinking, style="dim")
                    if content:
                        styled_text.append("\n\n")
                
                if content:
                    styled_text.append(content)
                
                return styled_text

            with Live(Text(""), console=console, refresh_per_second=20) as live:
                gen = agent.chat(user_input)
                try:
                    while True:
                        event = next(gen)
                        if event["type"] == "thinking":
                            full_thinking += event["chunk"]
                            live.update(render_response(full_thinking, full_content))
                        elif event["type"] == "content":
                            full_content += event["chunk"]
                            live.update(render_response(full_thinking, full_content))
                        elif event["type"] == "tool_call":
                            # Use a separate print or a special line in live for tool calls
                            live.console.print(f"[dim]Calling tool: {event['name']}...[/dim]")
                except StopIteration:
                    pass

            console.print()  # Final newline
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
