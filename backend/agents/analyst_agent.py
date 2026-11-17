
import json

from backend.services.llm_service import LLMService


class AnalystAgent:

    def __init__(self, llm: LLMService):
        self.llm = llm
        self.name = "Data Analyst Agent"

    def run(
        self,
        question: str,
        plan: str,
        research: str,
        dataset_context: str = ""
    ) -> str:

        # -----------------------------------------
        # 1. PREPARE DATASET CONTEXT
        # -----------------------------------------

        if dataset_context and dataset_context.strip():

            try:
                parsed_data = json.loads(dataset_context)

                dataset_summary = {
                    "dataset_name": parsed_data.get("dataset_name"),
                    "rows": parsed_data.get("rows"),
                    "columns": parsed_data.get("columns"),
                    "business_kpis": parsed_data.get("business_kpis", {}),
                    "regional_performance": parsed_data.get(
                        "regional_performance", []
                    ),
                    "product_performance": parsed_data.get(
                        "product_performance", []
                    ),
                    "monthly_trends": parsed_data.get(
                        "monthly_trends", []
                    ),
                    "business_insights": parsed_data.get(
                        "business_insights", []
                    ),
                    "data_quality_warnings": parsed_data.get(
                        "data_quality_warnings", []
                    ),
                    "numeric_summary": parsed_data.get(
                        "numeric_summary", {}
                    )
                }

                dataset_context = json.dumps(
                    dataset_summary,
                    indent=2,
                    ensure_ascii=False
                )

            except (json.JSONDecodeError, TypeError):
                # Preserve context if it is not valid JSON.
                pass

        else:
            dataset_context = "No dataset provided."

        # -----------------------------------------
        # 2. ANALYST SYSTEM INSTRUCTIONS
        # -----------------------------------------

        system_prompt = """
You are the Senior Business Intelligence Analyst
Agent of NEXUS AI, an autonomous multi-agent
business intelligence platform.

Your responsibility is to convert business data
and research findings into accurate, actionable
business intelligence.

ANALYSIS RESPONSIBILITIES:

1. FINANCIAL PERFORMANCE
   - Evaluate total revenue.
   - Evaluate total costs.
   - Evaluate total profit.
   - Interpret profit margins.
   - Identify financial performance concerns.

2. REGIONAL PERFORMANCE
   - Compare revenue across regions.
   - Identify the highest and lowest performers.
   - Explain possible business implications.

3. PRODUCT PERFORMANCE
   - Identify products generating the highest revenue.
   - Identify products generating the highest profit,
     when actual profit values are available.
   - Identify underperforming products when evidence
     supports that conclusion.

4. REVENUE TRENDS
   - Interpret monthly revenue trends when available.
   - Identify increases or decreases.
   - Explain the possible implications.

5. BUSINESS RISKS
   - Identify weak performance areas.
   - Evaluate cost and profitability concerns.
   - Identify data-quality limitations.

6. BUSINESS OPPORTUNITIES
   - Identify opportunities for revenue growth.
   - Recommend ways to improve profitability.
   - Suggest operational improvements.

STRICT DATA ACCURACY RULES:

- Use only numerical values supplied in the dataset context.
- Never invent revenue, profit, costs, percentages,
  rankings, growth rates or other numerical results.
- Do not calculate alternative figures when the
  supplied calculated KPI is available.
- Clearly distinguish actual findings from assumptions.
- Do not claim that a trend exists if the required
  time-series data is unavailable.
- Do not describe a product as most profitable if
  only its revenue is available.
- If a field is unavailable, explicitly state that
  the analysis cannot determine it.
- Treat missing data and quality warnings as limitations.
- Research findings are supporting context, not
  proof of dataset performance.

OUTPUT FORMAT:

1. Executive Analysis
2. Financial Performance
3. Regional Performance
4. Product Performance
5. Revenue Trends
6. Key Business Risks
7. Growth Opportunities
8. Strategic Recommendations
9. KPIs to Monitor
10. Data Limitations

For every important finding:
- State the actual evidence.
- Explain its business meaning.
- Give a practical recommendation where appropriate.

Write in professional business language.
Avoid generic recommendations that do not relate
to the available business evidence.
"""

        # -----------------------------------------
        # 3. ANALYST TASK
        # -----------------------------------------

        prompt = f"""
BUSINESS CHALLENGE:
{question}

PLANNER AGENT OUTPUT:
{plan}

RESEARCH AGENT OUTPUT:
{research}

CALCULATED DATASET ANALYSIS:
{dataset_context}

YOUR TASK:

Perform a detailed business intelligence analysis
using the information supplied above.

Prioritize actual dataset calculations over general
assumptions.

Identify the strongest and weakest business areas.

Explain what the findings mean for management.

Provide practical recommendations supported by
the available evidence.

Clearly identify information that is unavailable
rather than guessing.

Do not fabricate any numerical values.
"""

        # -----------------------------------------
        # 4. GENERATE ANALYSIS
        # -----------------------------------------

        return self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt
        )
