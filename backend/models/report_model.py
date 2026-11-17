
from datetime import datetime, timezone
from bson import ObjectId


def create_report_document(
    title: str,
    business_question: str,
    report_content: str,
    workflow_status: str = "completed",
    dataset_attached: bool = False
):
    """Create a MongoDB document for an executive report."""

    return {
        "title": title,
        "business_question": business_question,
        "report_content": report_content,
        "workflow_status": workflow_status,
        "dataset_attached": dataset_attached,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }


def serialize_report(report):
    """Convert MongoDB report data into JSON-safe data."""

    if not report:
        return None

    return {
        "id": str(report["_id"]),
        "title": report.get("title"),
        "business_question": report.get("business_question"),
        "report_content": report.get("report_content"),
        "workflow_status": report.get("workflow_status"),
        "dataset_attached": report.get("dataset_attached", False),
        "created_at": (
            report["created_at"].isoformat()
            if report.get("created_at") else None
        ),
        "updated_at": (
            report["updated_at"].isoformat()
            if report.get("updated_at") else None
        )
    }
