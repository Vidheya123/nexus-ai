
import json

from backend.services.llm_service import LLMService


class ResearchAgent:

    def __init__(self, llm: LLMService):
        self.llm = llm
        self.name = "Research Agent"

    def run(
        self,
        question: str,
        plan: str,
        dataset_context: str = "",
        tool_results: dict | None = None
    ) -> str:

        tool_results = tool_results or {}

        # -----------------------------------------
        # 1. PREPARE AVAILABLE EVIDENCE
        # -----------------------------------------

        dataset_available = bool(
            dataset_context and dataset_context.strip()
        )

        if dataset_available:
            try:
                parsed_dataset = json.loads(dataset_context)

                dataset_evidence = json.dumps(
                    {
                        "dataset_name": parsed_dataset.get("dataset_name"),
                        "rows": parsed_dataset.get("rows"),
                        "columns": parsed_dataset.get("columns"),
                        "business_kpis": parsed_dataset.get(
                            "business_kpis", {}
                        ),
                        "regional_performance": parsed_dataset.get(
                            "regional_performance", []
                        ),
                        "product_performance": parsed_dataset.get(
                            "product_performance", []
                        ),
                        "monthly_trends": parsed_dataset.get(
                            "monthly_trends", []
                        ),
                        "business_insights": parsed_dataset.get(
                            "business_insights", []
                        ),
                        "data_quality_warnings": parsed_dataset.get(
                            "data_quality_warnings", []
                        )
                    },
                    indent=2,
                    ensure_ascii=False
                )

            except (json.JSONDecodeError, TypeError):
                dataset_evidence = (
                    "Dataset was supplied, but its JSON format "
                    "could not be interpreted."
                )
        else:
            dataset_evidence = "No dataset was attached."

        tool_evidence = json.dumps(
            tool_results,
            indent=2,
            default=str
        )

        # -----------------------------------------
        # 2. RESEARCH AGENT INSTRUCTIONS
        # -----------------------------------------

        system_prompt = """
You are the Research Agent of NEXUS AI.

Your responsibility is to provide strategic business
context that supports the user's business question.

You work alongside:
- Planner Agent
- Autonomous Tool Agent
- Data Analyst Agent
- Report Generator Agent

YOUR RESPONSIBILITIES:

1. Identify relevant business and operational factors.
2. Explain possible market and industry considerations.
3. Identify potential customer and competitive factors.
4. Identify possible business opportunities.
5. Identify operational improvement opportunities.
6. Provide useful strategic context for the Data Analyst.

STRICT EVIDENCE RULES:

1. Dataset values are supplied separately.
   Do not ask the user to provide figures that are
   already available in the dataset.

2. Do not recalculate or alter supplied business KPIs.

3. Do not invent numerical statistics, market sizes,
   growth rates, industry averages or benchmarks.

4. Do not invent citations, research papers, URLs,
   companies' financial results or sources.

5. This agent does not have live web search.
   Never claim that current market research was performed.

6. Clearly distinguish:
   - Verified dataset findings
   - General business knowledge
   - Potential business opportunities
   - Information that is unavailable

7. Do not present general knowledge as a verified
   fact about the user's organization.

8. If the dataset does not contain a particular
   business dimension, identify it as unavailable.

9. Do not repeat the financial analysis already
   performed by the Data Analyst Agent.

10. Do not request financial figures when they
    are already present in the supplied evidence.

OUTPUT STRUCTURE:

1. Research Objective
2. Relevant Business Factors
3. Market and Competitive Considerations
4. Customer and Operational Considerations
5. Potential Business Opportunities
6. Strategic Considerations
7. Information Gaps and Research Limitations

Write in professional business language.
Keep the findings relevant to the original question.
"""

        # -----------------------------------------
        # 3. RESEARCH TASK
        # -----------------------------------------

        prompt = f"""
BUSINESS QUESTION:
{question}

PLANNER'S EXECUTION PLAN:
{plan}

DATASET ATTACHED:
{dataset_available}

AVAILABLE DATASET EVIDENCE:
{dataset_evidence}

AUTONOMOUS TOOL EXECUTION RESULTS:
{tool_evidence}

YOUR TASK:

Investigate the business challenge using the
available information.

The dataset and tool results are supplied so you
understand what has already been established.

Do not ask for figures that are already available.

Do not invent market statistics or external sources.

Focus on strategic context, business opportunities,
operational considerations and information gaps.

Clearly label general business knowledge as
general context rather than verified company data.
"""

        # -----------------------------------------
        # 4. GENERATE RESEARCH
        # -----------------------------------------

        return self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt
        )
