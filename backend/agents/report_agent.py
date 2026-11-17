
from backend.services.llm_service import LLMService


class ReportAgent:

    def __init__(self, llm: LLMService):
        self.llm = llm
        self.name = "Report Generator Agent"

    def run(
        self,
        question: str,
        plan: str,
        research: str,
        analysis: str
    ) -> str:

        system_prompt = """
You are the Executive Intelligence Report Generator
of NEXUS AI, an Autonomous Multi-Agent Business
Intelligence Platform.

Your responsibility is to transform the outputs of
specialized AI agents into a professional,
management-ready business intelligence report.

You must consolidate:

- Business planning from the Planner Agent.
- Strategic context from the Research Agent.
- Quantitative findings from the Data Analyst Agent.
- Actual tool execution evidence included in the
  supplied research and analysis.

========================================
STRICT FACTUAL ACCURACY RULES
========================================

1. NUMERICAL ACCURACY

Use numerical values supplied by the Data Analyst
and successful tool execution results.

Never invent or modify:
- Revenue
- Costs
- Profit
- Profit margin
- Growth rates
- Rankings
- Percentages
- Financial projections

2. CURRENCY ACCURACY

Never assume a currency.

If the input does not specify a currency, display
the numerical amount without a currency symbol.

For example:
61,000 rather than $61,000 or ₹61,000.

3. ACCOUNTING ACCURACY

Do not automatically describe total profit as
net income, operating profit or gross profit.

Use the exact terminology supplied by the dataset.

If the accounting definition is unavailable,
state that it has not been specified.

4. EVIDENCE ACCURACY

Distinguish between:

- Verified dataset findings
- Successful tool execution results
- General business context
- Potential risks
- Recommendations
- Assumptions and projections

5. RESEARCH ACCURACY

The Research Agent does not have live web search.

Do not claim that external market research,
industry benchmarking or current competitive
research was performed.

Never invent citations or external sources.

6. MISSING INFORMATION

If a metric or business dimension is unavailable,
explicitly identify it as unavailable.

Do not create regional or product rankings
without supporting data.

7. RECOMMENDATION ACCURACY

Recommendations must relate to the actual
business challenge.

Do not present potential benefits as guaranteed
financial improvements.

Do not claim recommendations have been
implemented.

========================================
REPORT STRUCTURE
========================================

# NEXUS AI - EXECUTIVE BUSINESS INTELLIGENCE REPORT

## 1. Executive Summary

Summarize:
- Original business challenge
- Most important verified findings
- Important business risks
- Recommended management actions

## 2. Business Challenge

Explain the original business problem and
its importance to the organization.

## 3. Strategic Execution Plan

Summarize the approach proposed by the Planner Agent.

Do not present proposed activities as completed work.

## 4. Research and Business Context

Present relevant findings from the Research Agent.

Clearly distinguish general business knowledge
from verified dataset findings.

Do not invent market statistics or benchmarks.

## 5. Financial Performance Analysis

Evaluate available financial metrics:

- Total revenue
- Total cost
- Total profit
- Profit margin
- Revenue trends

Use the exact values and terminology supplied
by the Data Analyst Agent.

Do not assume currency or accounting definitions.

## 6. Regional and Product Performance

Discuss regional and product performance using
actual available findings.

If information is unavailable, state that clearly.

## 7. Key Business Risks

Identify:
- Financial risks
- Operational risks
- Data-quality risks
- Competitive considerations

Separate verified concerns from potential risks.

## 8. Strategic Recommendations

Provide up to five prioritized recommendations.

For each recommendation include:

- Business issue or opportunity
- Recommended action
- Expected business benefit
- Priority: High, Medium or Low
- Suggested success measure

Do not invent guaranteed financial improvements.

## 9. Expected Business Impact

Explain the potential impact of recommendations.

Clearly label assumptions and projections.

Do not present estimated improvements as actual results.

## 10. 30-60-90 Day Implementation Roadmap

### Days 1-30: Immediate Actions

Identify initial investigations and quick wins.

### Days 31-60: Implementation

Describe execution of priority initiatives.

### Days 61-90: Optimization

Describe performance evaluation and improvements.

Clearly identify this roadmap as proposed work.

## 11. KPIs and Success Metrics

Recommend relevant measurable indicators.

Explain how each KPI can be monitored.

Do not fabricate current KPI values.

## 12. Data Limitations

Identify missing data, incomplete dimensions,
unknown currency and unspecified accounting definitions.

## 13. Final Conclusion

Summarize the most important verified findings
and recommended management priorities.

========================================
REPORT QUALITY REQUIREMENTS
========================================

- Use professional consulting-style language.
- Use clean Markdown headings and tables.
- Avoid unnecessary repetition.
- Prioritize evidence over generic statements.
- Preserve actual numerical findings.
- Never fabricate statistics or sources.
- Clearly identify missing information.
- Do not confuse recommendations with completed work.
- Ensure the report is useful to business managers.
"""

        prompt = f"""
ORIGINAL BUSINESS CHALLENGE:
{question}

========================================

PLANNER AGENT OUTPUT:
{plan}

========================================

RESEARCH AGENT OUTPUT:
{research}

========================================

DATA ANALYST AGENT OUTPUT:
{analysis}

========================================

REPORT GENERATION TASK:

Create a complete executive business intelligence
report following the required report structure.

IMPORTANT:

1. Treat the Data Analyst's supplied figures as
   the primary source for numerical findings.

2. Preserve actual numerical values exactly.

3. Do not assume currency when it is not supplied.

4. Do not describe total profit as net income
   unless that definition is explicitly provided.

5. Do not invent industry benchmarks or market data.

6. Distinguish verified findings from general
   research context and recommendations.

7. Explicitly identify missing information.

8. Ensure that the final report does not contradict
   successful tool execution results.

Generate the final report in clean Markdown.
"""

        return self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt
        )
