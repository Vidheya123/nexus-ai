
import json
from typing import Any, Callable


class ToolRegistry:
    """Registers and executes tools available to NEXUS AI."""

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        function: Callable
    ):
        self.tools[name] = {
            "description": description,
            "parameters": parameters,
            "function": function
        }

    def get_tool_definitions(self) -> list[dict]:
        """Return tools in Groq/OpenAI function-calling format."""

        definitions = []

        for name, tool in self.tools.items():
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })

        return definitions

    def execute(self, name: str, arguments: dict) -> dict:
        """Execute a registered tool safely."""

        if name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {name}"
            }

        try:
            result = self.tools[name]["function"](**arguments)

            return {
                "success": True,
                "tool": name,
                "result": result
            }

        except Exception as error:
            return {
                "success": False,
                "tool": name,
                "error": str(error)
            }
