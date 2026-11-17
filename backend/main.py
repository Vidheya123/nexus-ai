
import uuid
import json
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pymongo.errors import PyMongoError

from backend.config import APP_NAME, GROQ_MODEL, DATA_DIR
from backend.database.mongodb import (
    connect_to_mongodb,
    close_mongodb_connection,
    get_collection
)
from backend.models.report_model import (
    create_report_document,
    serialize_report
)
from backend.services.llm_service import LLMService
from backend.services.workflow import NexusWorkflow
from backend.services.data_service import DataAnalysisService


# =====================================================
# DATABASE LIFECYCLE
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to MongoDB when the application starts."""

    print("Starting NEXUS AI...")

    try:
        connect_to_mongodb()

        # Create indexes for faster report retrieval.
        get_collection("reports").create_index(
            [("created_at", -1)]
        )

        get_collection("chat_history").create_index(
            [("created_at", -1)]
        )

        get_collection("analysis_results").create_index(
            [("created_at", -1)]
        )

        print("NEXUS AI database initialization completed.")

        yield

    finally:
        close_mongodb_connection()
        print("NEXUS AI shutdown completed.")


# =====================================================
# APPLICATION CONFIGURATION
# =====================================================

app = FastAPI(
    title=APP_NAME,
    description="Autonomous Multi-Agent Business Intelligence Platform",
    version="1.3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# REQUEST MODELS
# =====================================================

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    model: str


class AgentRequest(BaseModel):
    question: str = Field(min_length=5)
    dataset_context: str = ""


# =====================================================
# DATABASE HELPERS
# =====================================================

def make_json_safe(data):
    """Convert Python data into MongoDB-safe JSON data."""

    return json.loads(
        json.dumps(data, default=str)
    )


def save_generated_report(
    result: dict,
    question: str,
    dataset_attached: bool = False
):
    """Extract and save a generated executive report."""

    workflow = result.get("agent_workflow", result)

    report_content = (
        workflow.get("final_report")
        or result.get("final_report")
    )

    if not report_content:
        return None

    if not isinstance(report_content, str):
        report_content = json.dumps(
            report_content,
            indent=2,
            default=str
        )

    title = question.strip()[:100]

    document = create_report_document(
        title=title,
        business_question=question,
        report_content=report_content,
        workflow_status=workflow.get(
            "status",
            "completed"
        ),
        dataset_attached=dataset_attached
    )

    # Preserve the workflow execution information.
    document["agent_workflow"] = make_json_safe(workflow)

    saved = get_collection("reports").insert_one(document)

    return str(saved.inserted_id)


def save_agent_execution(
    question: str,
    result: dict,
    report_id: str = None
):
    """Save agent execution metadata."""

    workflow = result.get("agent_workflow", result)

    document = {
        "business_question": question,
        "agents_executed": workflow.get(
            "agents_executed", 0
        ),
        "workflow_status": workflow.get(
            "status", "unknown"
        ),
        "execution_log": make_json_safe(
            workflow.get("execution_log", [])
        ),
        "report_id": report_id,
        "created_at": datetime.now(timezone.utc)
    }

    get_collection("agent_executions").insert_one(document)


# =====================================================
# HOME ENDPOINT
# =====================================================

@app.get("/")
def home():
    return {
        "application": APP_NAME,
        "message": "Welcome to NEXUS AI",
        "status": "running",
        "version": "1.3.0"
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health_check():

    try:
        get_collection("reports").database.client.admin.command(
            "ping"
        )
        database_status = "connected"

    except Exception:
        database_status = "disconnected"

    return {
        "application": APP_NAME,
        "status": "healthy",
        "model": GROQ_MODEL,
        "backend": "FastAPI",
        "database": database_status
    }


# =====================================================
# BASIC AI CHAT ENDPOINT
# =====================================================

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        service = LLMService()

        answer = service.generate_response(
            prompt=request.question,
            system_prompt=(
                "You are NEXUS AI, an intelligent business "
                "intelligence assistant. Provide accurate, "
                "structured and actionable business insights."
            )
        )

        # Save conversation to MongoDB.
        chat_document = {
            "question": request.question,
            "answer": answer,
            "model": GROQ_MODEL,
            "created_at": datetime.now(timezone.utc)
        }

        get_collection("chat_history").insert_one(
            chat_document
        )

        return ChatResponse(
            answer=answer,
            model=GROQ_MODEL
        )

    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except PyMongoError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Chat generated, but database save failed: {error}"
        )

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"AI service error: {str(error)}"
        )


# =====================================================
# MULTI-AGENT WORKFLOW ENDPOINT
# =====================================================

@app.post("/api/agents/run")
def run_agents(request: AgentRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        workflow = NexusWorkflow()

        result = workflow.run(
            question=request.question,
            dataset_context=request.dataset_context
        )

        result["model"] = GROQ_MODEL

        # Save executive report if generated.
        report_id = save_generated_report(
            result=result,
            question=request.question,
            dataset_attached=False
        )

        # Save workflow execution history.
        save_agent_execution(
            question=request.question,
            result=result,
            report_id=report_id
        )

        result["report_id"] = report_id

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except PyMongoError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution completed, but database save failed: {error}"
        )

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Agent workflow failed: {str(error)}"
        )


# =====================================================
# CSV DATA ANALYSIS ENDPOINT
# =====================================================

@app.post("/api/data/analyze")
async def analyze_dataset(
    file: UploadFile = File(...)
):

    if (
        not file.filename
        or Path(file.filename).suffix.lower() != ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid CSV file."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 10 MB."
        )

    temp_path = DATA_DIR / f"{uuid.uuid4().hex}.csv"

    try:
        temp_path.write_bytes(content)

        analyzer = DataAnalysisService()

        result = analyzer.analyze_csv(
            str(temp_path)
        )

        # Save analysis results in MongoDB.
        analysis_document = {
            "filename": file.filename,
            "analysis": make_json_safe(result),
            "created_at": datetime.now(timezone.utc)
        }

        get_collection("analysis_results").insert_one(
            analysis_document
        )

        return {
            "status": "success",
            "message": "Dataset analyzed and saved successfully",
            "analysis": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except PyMongoError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset analyzed, but database save failed: {error}"
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset analysis failed: {str(error)}"
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()

        await file.close()


# =====================================================
# CSV + MULTI-AGENT INTELLIGENCE ENDPOINT
# =====================================================

@app.post("/api/agents/analyze-dataset")
async def analyze_dataset_with_agents(
    question: str = Form(...),
    file: UploadFile = File(...)
):

    if (
        not file.filename
        or Path(file.filename).suffix.lower() != ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 10 MB."
        )

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a business question."
        )

    temp_path = DATA_DIR / f"{uuid.uuid4().hex}.csv"

    try:
        # STEP 1: Save temporary CSV.
        temp_path.write_bytes(content)

        # STEP 2: Perform Pandas analysis.
        analyzer = DataAnalysisService()

        dataset_result = analyzer.analyze_csv(
            str(temp_path)
        )

        # STEP 3: Convert analysis into agent context.
        dataset_context = json.dumps(
            dataset_result,
            indent=2,
            default=str
        )

        # STEP 4: Execute multi-agent workflow.
        workflow = NexusWorkflow()

        agent_result = workflow.run(
            question=question,
            dataset_context=dataset_context
        )

        agent_result["model"] = GROQ_MODEL

        combined_result = {
            "status": "success",
            "message": "Multi-agent dataset analysis completed",
            "model": GROQ_MODEL,
            "dataset_analysis": dataset_result,
            "agent_workflow": agent_result
        }

        # STEP 5: Save dataset analysis.
        get_collection("analysis_results").insert_one({
            "filename": file.filename,
            "business_question": question,
            "analysis": make_json_safe(dataset_result),
            "created_at": datetime.now(timezone.utc)
        })

        # STEP 6: Save executive report.
        report_id = save_generated_report(
            result=combined_result,
            question=question,
            dataset_attached=True
        )

        # STEP 7: Save agent execution.
        save_agent_execution(
            question=question,
            result=combined_result,
            report_id=report_id
        )

        combined_result["report_id"] = report_id

        return combined_result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except PyMongoError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow completed, but database save failed: {error}"
        )

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Dataset agent workflow failed: {str(error)}"
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()

        await file.close()


# =====================================================
# REPORT HISTORY - GET ALL REPORTS
# =====================================================

@app.get("/api/reports")
def get_all_reports():

    try:
        reports = get_collection("reports").find(
            {},
            {
                "report_content": 0,
                "agent_workflow": 0
            }
        ).sort("created_at", -1)

        result = []

        for report in reports:
            result.append({
                "id": str(report["_id"]),
                "title": report.get("title"),
                "business_question": report.get(
                    "business_question"
                ),
                "workflow_status": report.get(
                    "workflow_status"
                ),
                "dataset_attached": report.get(
                    "dataset_attached", False
                ),
                "created_at": (
                    report["created_at"].isoformat()
                    if report.get("created_at") else None
                )
            })

        return {
            "status": "success",
            "count": len(result),
            "reports": result
        }

    except PyMongoError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve reports: {error}"
        )


# =====================================================
# GET INDIVIDUAL REPORT
# =====================================================

@app.get("/api/reports/{report_id}")
def get_report(report_id: str):

    try:
        object_id = ObjectId(report_id)

    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid report ID."
        )

    try:
        report = get_collection("reports").find_one({
            "_id": object_id
        })

        if not report:
            raise HTTPException(
                status_code=404,
                detail="Report not found."
            )

        return {
            "status": "success",
            "report": serialize_report(report)
        }

    except PyMongoError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve report: {error}"
        )


# =====================================================
# DELETE INDIVIDUAL REPORT
# =====================================================

@app.delete("/api/reports/{report_id}")
def delete_report(report_id: str):

    try:
        object_id = ObjectId(report_id)

    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid report ID."
        )

    try:
        result = get_collection("reports").delete_one({
            "_id": object_id
        })

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Report not found."
            )

        return {
            "status": "success",
            "message": "Report deleted successfully.",
            "report_id": report_id
        }

    except PyMongoError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete report: {error}"
        )


# =====================================================
# CHAT HISTORY - GET ALL CONVERSATIONS
# =====================================================

@app.get("/api/chat/history")
def get_chat_history():

    try:
        conversations = get_collection("chat_history").find(
            {},
            {"answer": 0}
        ).sort("created_at", -1)

        result = []

        for item in conversations:
            result.append({
                "id": str(item["_id"]),
                "question": item.get("question"),
                "model": item.get("model"),
                "created_at": (
                    item["created_at"].isoformat()
                    if item.get("created_at") else None
                )
            })

        return {
            "status": "success",
            "count": len(result),
            "conversations": result
        }

    except PyMongoError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve chat history: {error}"
        )
