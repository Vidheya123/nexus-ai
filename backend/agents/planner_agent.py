
from backend.services.llm_service import LLMService


class PlannerAgent:
    def __init__(self, llm: LLMService):
        self.llm = llm
        self.name = "Planner Agent"

    def run(self, question: str) -> str:
        system_prompt = """
        You are the Planner Agent of NEXUS AI.

        Your responsibility is to understand business problems
        and create structured execution plans.

        Identify:
        1. Business objective
        2. Key questions to investigate
        3. Research requirements
        4. Data analysis requirements
        5. Expected business outcomes

        Create a clear, numbered execution plan.

        Do not invent business data or claim that analysis
        has already been completed.
        """

        return self.llm.generate_response(
            prompt=question,
            system_prompt=system_prompt
        )
