
import json

from backend.services.tool_registry import ToolRegistry
from backend.services.llm_service import LLMService


class AutonomousToolAgent:

    def __init__(
        self,
        llm: LLMService,
        tool_registry: ToolRegistry,
        max_tool_calls: int = 5
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_tool_calls = max_tool_calls
        self.name = "Autonomous Tool Agent"

    def run(
        self,
        question: str,
        context: dict | None = None
    ) -> dict:

        context = context or {}

        tool_results = []
        execution_log = []
        summary = ""

        # -------------------------------------------------
        # 1. PREPARE CONVERSATION
        # -------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": """
You are the Autonomous Tool Agent of NEXUS AI.

You are responsible for solving business problems by
independently selecting and executing available tools.

YOUR RESPONSIBILITIES:

1. Understand the user's business question.
2. Inspect the available context.
3. Select the most relevant tools.
4. Execute tools when actual data is required.
5. Inspect the returned tool results.
6. Decide whether additional tools are necessary.
7. Generate a final summary using actual tool results.

AUTONOMOUS DECISION RULES:

- Select tools based on the user's actual requirements.
- You may execute multiple tools when necessary.
- Do not call the same tool repeatedly without a reason.
- Do not invent tools that are not registered.
- Do not invent tool execution results.
- Do not fabricate numerical business information.
- If a tool fails, acknowledge the failure.
- If no dataset is available, do not assume dataset values.
- Stop calling tools once sufficient information is available.

FINAL RESPONSE:

Provide a concise business intelligence summary.

Use actual tool outputs as evidence.
Clearly identify missing information.
Explain important business findings.
"""
            },
            {
                "role": "user",
                "content": (
                    f"BUSINESS QUESTION:\n{question}\n\n"
                    f"AVAILABLE CONTEXT:\n"
                    f"{json.dumps(context, indent=2, default=str)}"
                )
            }
        ]

        # -------------------------------------------------
        # 2. LOAD AVAILABLE TOOLS
        # -------------------------------------------------

        definitions = self.tool_registry.get_tool_definitions()

        if not definitions:
            return {
                "status": "completed",
                "agent": self.name,
                "tool_calls": [],
                "tool_results": [],
                "summary": "No tools are currently registered."
            }

        # -------------------------------------------------
        # 3. AUTONOMOUS TOOL EXECUTION LOOP
        # -------------------------------------------------

        for step in range(self.max_tool_calls):

            try:
                response = (
                    self.llm.client.chat.completions.create(
                        model=self.llm.model,
                        messages=messages,
                        tools=definitions,
                        tool_choice="auto",
                        temperature=0.2,
                        max_tokens=1500
                    )
                )

            except Exception as error:

                summary = (
                    "The autonomous agent encountered an "
                    f"LLM execution error: {str(error)}"
                )

                execution_log.append({
                    "step": step + 1,
                    "agent": self.name,
                    "status": "failed",
                    "error": str(error)
                })

                break

            assistant_message = response.choices[0].message

            # -------------------------------------------------
            # 4. CHECK WHETHER THE LLM REQUESTED TOOLS
            # -------------------------------------------------

            if not assistant_message.tool_calls:

                summary = assistant_message.content or (
                    "The agent completed without generating "
                    "a textual summary."
                )

                break

            # Add assistant tool-call message to conversation.
            messages.append(
                assistant_message.model_dump(exclude_none=True)
            )

            # -------------------------------------------------
            # 5. EXECUTE SELECTED TOOLS
            # -------------------------------------------------

            limit_reached = False

            for tool_call in assistant_message.tool_calls:

                # Enforce the maximum number of executions.
                if len(tool_results) >= self.max_tool_calls:
                    limit_reached = True
                    break

                tool_name = tool_call.function.name

                # Parse tool arguments.
                try:

                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Tool arguments must be a JSON object."
                        )

                except (json.JSONDecodeError, ValueError) as error:

                    result = {
                        "success": False,
                        "tool": tool_name,
                        "error": (
                            f"Invalid tool arguments: {str(error)}"
                        )
                    }

                else:

                    # Execute the selected registered tool.
                    result = self.tool_registry.execute(
                        tool_name,
                        arguments
                    )

                # Save actual tool result.
                tool_results.append(result)

                # Record execution details.
                execution_log.append({
                    "step": step + 1,
                    "agent": self.name,
                    "tool": tool_name,
                    "status": (
                        "completed"
                        if result.get("success")
                        else "failed"
                    )
                })

                # Send tool output back to Groq.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        default=str
                    )
                })

            if limit_reached:
                summary = (
                    "Maximum tool execution limit reached. "
                    "The available results are returned below."
                )
                break

            if len(tool_results) >= self.max_tool_calls:
                summary = (
                    "Maximum tool execution limit reached. "
                    "The available results are returned below."
                )
                break

        else:

            summary = (
                "Maximum tool-call rounds reached. "
                "The available results are returned below."
            )

        # -------------------------------------------------
        # 6. FINAL RESPONSE
        # -------------------------------------------------

        return {
            "status": "completed",
            "agent": self.name,
            "question": question,
            "total_tool_calls": len(tool_results),
            "tool_calls": execution_log,
            "tool_results": tool_results,
            "summary": summary
        }
