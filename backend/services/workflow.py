
import json

from backend.services.llm_service import LLMService
from backend.services.tool_registry import ToolRegistry

from backend.agents.planner_agent import PlannerAgent
from backend.agents.research_agent import ResearchAgent
from backend.agents.analyst_agent import AnalystAgent
from backend.agents.report_agent import ReportAgent
from backend.agents.autonomous_tool_agent import AutonomousToolAgent


class NexusWorkflow:

    def __init__(self):
        self.llm = LLMService()

        # Existing specialized agents
        self.planner = PlannerAgent(self.llm)
        self.researcher = ResearchAgent(self.llm)
        self.analyst = AnalystAgent(self.llm)
        self.reporter = ReportAgent(self.llm)

        print("NEXUS AI workflow initialized.")

    # -------------------------------------------------
    # 1. DATASET CONTEXT
    # -------------------------------------------------

    def _get_dataset(self, dataset_context: str) -> dict:

        if not dataset_context or not dataset_context.strip():
            raise ValueError("No dataset is attached to this request.")

        try:
            data = json.loads(dataset_context)

        except json.JSONDecodeError as error:
            raise ValueError(
                "Dataset context is not valid JSON."
            ) from error

        if not isinstance(data, dict):
            raise ValueError(
                "Dataset context must be a JSON object."
            )

        return data

    # -------------------------------------------------
    # 2. DATASET TOOLS
    # -------------------------------------------------

    def _get_dataset_summary(self, dataset_context: str) -> dict:

        data = self._get_dataset(dataset_context)

        return {
            "dataset_name": data.get("dataset_name"),
            "rows": data.get("rows"),
            "columns": data.get("columns"),
            "column_names": data.get("column_names", []),
            "duplicate_rows": data.get("duplicate_rows"),
            "missing_values": data.get("missing_values", {}),
            "data_quality_warnings": data.get(
                "data_quality_warnings", []
            )
        }

    def _get_business_kpis(self, dataset_context: str) -> dict:

        data = self._get_dataset(dataset_context)

        kpis = data.get("business_kpis", {})

        if not kpis:
            return {
                "message": "No calculated business KPIs are available."
            }

        return kpis

    def _get_regional_performance(
        self,
        dataset_context: str
    ) -> dict:

        data = self._get_dataset(dataset_context)

        regional_data = data.get("regional_performance", [])

        if not regional_data:
            return {
                "message": "Regional performance data is unavailable."
            }

        return {
            "regional_performance": regional_data
        }

    # -------------------------------------------------
    # 3. CREATE REQUEST-SPECIFIC TOOL REGISTRY
    # -------------------------------------------------

    def _create_tool_registry(
        self,
        dataset_context: str
    ) -> ToolRegistry:

        registry = ToolRegistry()

        # Tool 1: Dataset Summary
        registry.register(
            name="get_dataset_summary",
            description=(
                "Get the dataset name, row count, column count, "
                "available columns and data quality information."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            function=lambda: self._get_dataset_summary(
                dataset_context
            )
        )

        # Tool 2: Business KPIs
        registry.register(
            name="get_business_kpis",
            description=(
                "Retrieve actual calculated business KPIs, "
                "including revenue, costs, profit and profit margin."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            function=lambda: self._get_business_kpis(
                dataset_context
            )
        )

        # Tool 3: Regional Performance
        registry.register(
            name="get_regional_performance",
            description=(
                "Retrieve regional revenue and profit performance "
                "to identify the highest and lowest performing regions."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            function=lambda: self._get_regional_performance(
                dataset_context
            )
        )

        return registry

    # -------------------------------------------------
    # 4. MAIN WORKFLOW
    # -------------------------------------------------

    def run(
        self,
        question: str,
        dataset_context: str = ""
    ) -> dict:

        execution_log = []

        # -------------------------------------------------
        # AGENT 1: PLANNER
        # -------------------------------------------------

        plan = self.planner.run(question)

        execution_log.append({
            "agent": "Planner Agent",
            "status": "completed",
            "output": plan
        })

        # -------------------------------------------------
        # AGENT 2: AUTONOMOUS TOOL AGENT
        # -------------------------------------------------

        if dataset_context and dataset_context.strip():

            tool_registry = self._create_tool_registry(
                dataset_context
            )

            autonomous_agent = AutonomousToolAgent(
                llm=self.llm,
                tool_registry=tool_registry,
                max_tool_calls=5
            )

            tool_context = {
                "dataset_attached": True,
                "available_dataset_information": (
                    "A dataset is attached. Select relevant "
                    "business intelligence tools."
                ),
                "execution_plan": plan
            }

            autonomous_result = autonomous_agent.run(
                question=question,
                context=tool_context
            )

        else:

            autonomous_result = {
                "status": "skipped",
                "agent": "Autonomous Tool Agent",
                "total_tool_calls": 0,
                "tool_calls": [],
                "tool_results": [],
                "summary": (
                    "No dataset is attached. "
                    "Dataset tools were not executed."
                )
            }

        execution_log.append({
            "agent": "Autonomous Tool Agent",
            "status": autonomous_result.get(
                "status", "completed"
            ),
            "output": autonomous_result
        })

        # -------------------------------------------------
        # AGENT 3: RESEARCH
        # -------------------------------------------------

        research = self.researcher.run(
    question=question,
    plan=plan,
    dataset_context=dataset_context,
    tool_results=autonomous_result
)

        execution_log.append({
            "agent": "Research Agent",
            "status": "completed",
            "output": research
        })

        # -------------------------------------------------
        # AGENT 4: DATA ANALYST
        # -------------------------------------------------

        enriched_research = f"""
RESEARCH AGENT FINDINGS:

{research}

AUTONOMOUS TOOL EXECUTION RESULTS:

{json.dumps(autonomous_result, indent=2, default=str)}

INSTRUCTIONS:

Use successful tool results as supporting evidence.

Do not treat failed tool calls as successful findings.

Do not invent numerical values.

Clearly identify unavailable information.
"""

        analysis = self.analyst.run(
            question=question,
            plan=plan,
            research=enriched_research,
            dataset_context=dataset_context
        )

        execution_log.append({
            "agent": "Data Analyst Agent",
            "status": "completed",
            "output": analysis
        })

        # -------------------------------------------------
        # AGENT 5: REPORT GENERATOR
        # -------------------------------------------------

        report = self.reporter.run(
            question=question,
            plan=plan,
            research=enriched_research,
            analysis=analysis
        )

        execution_log.append({
            "agent": "Report Generator Agent",
            "status": "completed",
            "output": report
        })

        # -------------------------------------------------
        # 6. FINAL WORKFLOW RESPONSE
        # -------------------------------------------------

        return {
            "question": question,
            "status": "completed",
            "agents_executed": len(execution_log),
            "dataset_attached": bool(dataset_context),
            "autonomous_tool_execution": autonomous_result,
            "execution_log": execution_log,
            "final_report": report
        }
