import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import re
import html
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable
)
# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="NEXUS AI | Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
API_URL = "http://127.0.0.1:8000"
# =====================================================
# SESSION STATE
# =====================================================
defaults = {
    "chat_history": [],
    "agent_result": None,
    "dataset_result": None,
    "dataset_df": None,
    "selected_saved_report_id": None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value
# =====================================================
# CUSTOM DESIGN SYSTEM
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --navy: #14213d;
    --blue: #3563e9;
    --purple: #7657e8;
    --text: #18243b;
    --muted: #68758b;
    --border: #e7ebf2;
    --surface: #ffffff;
    --background: #f5f7fb;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: var(--background);
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background: #111d35;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #e8edfa !important;
}
[data-testid="stSidebar"] hr {
    border-color: #34415b;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #1c2b49;
    border: 1px solid #354461;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: white !important;
}
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, h4, p, label,
[data-testid="stMarkdownContainer"] {
    color: var(--text);
}
h1 {
    font-weight: 800 !important;
    letter-spacing: -1px;
}
h2, h3 {
    font-weight: 700 !important;
}
p {
    line-height: 1.65;
}
.hero-tag {
    color: #c7d7ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero-title {
    color: white !important;
    font-size: 42px;
    font-weight: 800;
    margin: 12px 0;
}
.hero-description {
    color: #d6e2ff !important;
    font-size: 15px;
    max-width: 700px;
}
.hero {
    background: linear-gradient(120deg, #14213d, #263f79 65%, #5944a9);
    border-radius: 18px;
    padding: 36px;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(25, 44, 88, 0.13);
}
.section-heading {
    font-size: 21px;
    font-weight: 750;
    color: #18243b;
    margin: 22px 0 5px;
}
.section-subtitle {
    font-size: 13px;
    color: #68758b;
    margin-bottom: 18px;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    padding: 19px;
    border-radius: 13px;
    box-shadow: 0 3px 12px rgba(20, 40, 80, 0.035);
}
[data-testid="stMetricLabel"] {
    color: #68758b !important;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #18243b !important;
    font-weight: 800;
}
.stButton > button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 43px;
    border: 1px solid #dce3ef;
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(110deg, #3563e9, #7657e8);
    color: white;
    border: none;
}
.stButton > button:hover {
    border-color: #3563e9;
    transform: translateY(-1px);
}
.stTextArea textarea,
.stTextInput input {
    border-radius: 10px;
    border: 1px solid #dce3ef;
    background: white;
    color: #18243b;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 650;
    color: #68758b;
    padding: 13px 20px;
}
.stTabs [aria-selected="true"] {
    color: #3563e9 !important;
}
[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
}
[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #b9c7df;
    border-radius: 12px;
    padding: 15px;
}
.footer {
    text-align: center;
    padding: 25px;
    color: #8290a6;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)
# =====================================================
# PDF REPORT GENERATION
# =====================================================
def generate_pdf_report(report_text):
    """
    Convert the generated Markdown-style executive report
    into a downloadable PDF document.
    """
    pdf_buffer = BytesIO()
    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title="NEXUS AI Executive Report",
        author="NEXUS AI"
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="NexusTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#14213d"),
        alignment=TA_CENTER,
        spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name="NexusSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#68758b"),
        alignment=TA_CENTER,
        spaceAfter=15
    ))
    styles.add(ParagraphStyle(
        name="NexusH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#14213d"),
        spaceBefore=13,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="NexusH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#3563e9"),
        spaceBefore=10,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="NexusH3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#263f79"),
        spaceBefore=8,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="NexusBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#263247"),
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="NexusBullet",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        leftIndent=14,
        firstLineIndent=-8,
        textColor=colors.HexColor("#263247"),
        spaceAfter=4
    ))
    story = [
        Paragraph("NEXUS AI", styles["NexusTitle"]),
        Paragraph(
            "BUSINESS INTELLIGENCE &amp; STRATEGIC REPORT",
            styles["NexusSubtitle"]
        ),
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#dce3ef"),
            spaceAfter=12
        )
    ]
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        if line in ("---", "***", "___"):
            story.append(HRFlowable(
                width="100%",
                thickness=0.6,
                color=colors.HexColor("#dce3ef"),
                spaceBefore=5,
                spaceAfter=8
            ))
            continue
        # Escape text before adding the limited Markdown formatting.
        safe_line = html.escape(line, quote=False)
        safe_line = re.sub(
            r"\*\*(.+?)\*\*",
            r"<b>\1</b>",
            safe_line
        )
        safe_line = re.sub(
            r"\*(.+?)\*",
            r"<i>\1</i>",
            safe_line
        )
        safe_line = re.sub(
            r"`(.+?)`",
            r"<font name='Courier'>\1</font>",
            safe_line
        )
        if line.startswith("### "):
            story.append(Paragraph(safe_line[4:], styles["NexusH3"]))
        elif line.startswith("## "):
            story.append(Paragraph(safe_line[3:], styles["NexusH2"]))
        elif line.startswith("# "):
            story.append(Paragraph(safe_line[2:], styles["NexusH1"]))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph(
                "&#8226; " + safe_line[2:],
                styles["NexusBullet"]
            ))
        elif re.match(r"^\d+[.)]\s+", line):
            story.append(Paragraph(safe_line, styles["NexusBullet"]))
        else:
            story.append(Paragraph(safe_line, styles["NexusBody"]))
    def add_page_footer(canvas, doc):
        canvas.saveState()
        page_width, _ = A4
        canvas.setStrokeColor(colors.HexColor("#e7ebf2"))
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, 14 * mm, page_width - 20 * mm, 14 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#68758b"))
        canvas.drawString(20 * mm, 9 * mm, "NEXUS AI | Business Intelligence")
        canvas.drawRightString(
            page_width - 20 * mm,
            9 * mm,
            f"Page {doc.page}"
        )
        canvas.restoreState()
    document.build(
        story,
        onFirstPage=add_page_footer,
        onLaterPages=add_page_footer
    )
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
# =====================================================
# BACKEND CONNECTION
# =====================================================
@st.cache_data(ttl=10)
def get_backend_status():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None
backend = get_backend_status()
# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("# 🧠 NEXUS AI")
    st.caption("INTELLIGENCE PLATFORM")
    st.divider()
    st.markdown("### SYSTEM STATUS")
    if backend:
        st.success("● All systems operational")
        st.caption(f"Model: {backend.get('model', 'Groq')}")
    else:
        st.error("● Backend disconnected")
    st.divider()
    st.markdown("### PLATFORM")
    st.markdown("""
    **01**  AI Assistant
    **02**  Data Intelligence
    **03**  Agent Orchestration
    **04**  Executive Reports

    **05**  Report History
    """)
    st.divider()
    st.markdown("### TECHNOLOGY")
    st.caption("Python · FastAPI · Groq")
    st.caption("Pandas · Plotly · Streamlit")
    st.divider()
    st.caption("NEXUS AI • VERSION 1.3.0")
# =====================================================
# HERO SECTION
# =====================================================
st.html("""
<div class="hero">
    <div class="hero-tag">NEXT GENERATION BUSINESS INTELLIGENCE</div>
    <div class="hero-title">🧠 NEXUS AI</div>
    <div class="hero-description">
        Your intelligent business decision platform.
        Analyze data, orchestrate AI agents and transform
        complex business challenges into actionable insights.
    </div>
</div>
""")
# =====================================================
# PLATFORM OVERVIEW
# =====================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        "Platform Status",
        "Online" if backend else "Offline"
    )
with c2:
    st.metric("AI Engine", "Groq")
with c3:
    st.metric("AI Agents", "04")
with c4:
    st.metric("Analytics Engine", "Pandas")
st.markdown("<br>", unsafe_allow_html=True)
# =====================================================
# WORKSPACE NAVIGATION
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✦ AI Workspace",
    "▦ Data Intelligence",
    "◈ Agent Studio",
    "▤ Executive Reports",
    "◷ Report History"
])
# =====================================================
# TAB 1 - AI ASSISTANT
# =====================================================
with tab1:
    st.markdown(
        '<div class="section-heading">AI Business Assistant</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Ask complex business questions and receive intelligent recommendations.</div>',
        unsafe_allow_html=True
    )
    question = st.text_area(
        "What would you like NEXUS AI to investigate?",
        placeholder="Example: How can a retail company increase revenue and reduce operating costs?",
        height=125,
        key="chat_input"
    )
    if st.button("✦ Generate Business Insights", type="primary"):
        if not question.strip():
            st.warning("Please enter a business question.")
        elif not backend:
            st.error("Start the FastAPI backend first.")
        else:
            try:
                with st.spinner("NEXUS AI is thinking..."):
                    response = requests.post(
                        f"{API_URL}/api/chat",
                        json={"question": question},
                        timeout=120
                    )
                    response.raise_for_status()
                    result = response.json()
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "model": result["model"]
                })
                st.success("Intelligence generated successfully.")
            except requests.exceptions.RequestException as error:
                st.error(f"Request failed: {error}")
    for item in reversed(st.session_state.chat_history):
        with st.container(border=True):
            st.markdown("**YOUR QUESTION**")
            st.write(item["question"])
            st.divider()
            st.markdown("### ✦ NEXUS AI INSIGHTS")
            st.markdown(item["answer"])
            st.caption(f"Generated using {item['model']}")
# =====================================================
# TAB 2 - DATA INTELLIGENCE
# =====================================================
with tab2:
    st.markdown(
        '<div class="section-heading">Business Data Intelligence</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Upload your business dataset and discover performance insights.</div>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Choose your CSV dataset",
        type=["csv"],
        key="data_upload"
    )
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.dataset_df = df
            st.markdown("### Dataset Preview")
            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )
            if st.button("▦ Analyze Business Dataset", type="primary"):
                try:
                    with st.spinner("Calculating business KPIs..."):
                        response = requests.post(
                            f"{API_URL}/api/data/analyze",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "text/csv"
                                )
                            },
                            timeout=120
                        )
                        response.raise_for_status()
                        st.session_state.dataset_result = (
                            response.json()["analysis"]
                        )
                    st.success("Dataset analyzed successfully.")
                except requests.exceptions.RequestException as error:
                    st.error(f"Analysis failed: {error}")
            analysis = st.session_state.dataset_result
            if analysis:
                kpis = analysis.get("business_kpis", {})
                revenue = kpis.get("total_revenue")
                cost = kpis.get("total_cost")
                profit = kpis.get(
                    "total_profit",
                    kpis.get("calculated_profit")
                )
                margin = kpis.get("profit_margin_percentage")
                if margin is None and revenue and profit is not None:
                    margin = profit / revenue * 100
                def currency(value):
                    return f"₹{value:,.0f}" if value is not None else "N/A"
                st.divider()
                st.markdown("### Financial Performance")
                a, b, c, d = st.columns(4)
                a.metric("Revenue", currency(revenue))
                b.metric("Total Cost", currency(cost))
                c.metric("Net Profit", currency(profit))
                d.metric(
                    "Profit Margin",
                    f"{margin:.2f}%" if margin is not None else "N/A"
                )
                st.markdown("### Financial Analytics")
                metrics = []
                for name, value in [
                    ("Revenue", revenue),
                    ("Cost", cost),
                    ("Profit", profit)
                ]:
                    if value is not None:
                        metrics.append({
                            "Metric": name,
                            "Amount": value
                        })
                if metrics:
                    fig = px.bar(
                        pd.DataFrame(metrics),
                        x="Metric",
                        y="Amount",
                        color="Metric",
                        text_auto=".2s",
                        color_discrete_sequence=[
                            "#3563e9",
                            "#8b9bb8",
                            "#16a879"
                        ]
                    )
                    fig.update_layout(
                        template="plotly_white",
                        height=380,
                        showlegend=False,
                        margin=dict(l=20, r=20, t=25, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                # Regional performance
                original_df = st.session_state.dataset_df
                if original_df is not None:
                    cols = {
                        col.lower().strip(): col
                        for col in original_df.columns
                    }
                    if "region" in cols and "revenue" in cols:
                        region_col = cols["region"]
                        revenue_col = cols["revenue"]
                        regional = original_df.copy()
                        regional[revenue_col] = pd.to_numeric(
                            regional[revenue_col],
                            errors="coerce"
                        )
                        regional = (
                            regional.groupby(region_col)[revenue_col]
                            .sum()
                            .reset_index()
                        )
                        st.markdown("### Regional Revenue")
                        fig = px.bar(
                            regional,
                            x=region_col,
                            y=revenue_col,
                            color=region_col,
                            text_auto=".2s",
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        fig.update_layout(
                            template="plotly_white",
                            height=350,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Data Quality")
                q1, q2, q3 = st.columns(3)
                q1.metric("Rows", analysis.get("rows", 0))
                q2.metric("Columns", analysis.get("columns", 0))
                q3.metric(
                    "Duplicates",
                    analysis.get("duplicate_rows", 0)
                )
                with st.expander("View detailed data statistics"):
                    st.json(analysis.get("numeric_summary", {}))
        except Exception as error:
            st.error(f"Could not process the dataset: {error}")
# =====================================================
# TAB 3 - AGENT STUDIO
# =====================================================
with tab3:
    st.markdown(
        '<div class="section-heading">Multi-Agent Orchestration Studio</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Coordinate four specialized AI agents to solve complex business challenges.</div>',
        unsafe_allow_html=True
    )
    agent_descriptions = [
        ("01", "Planner Agent", "Business problem decomposition"),
        ("02", "Research Agent", "Strategic research and opportunities"),
        ("03", "Data Analyst Agent", "Metrics and performance analysis"),
        ("04", "Report Generator", "Executive intelligence report")
    ]
    for number, name, description in agent_descriptions:
        with st.container(border=True):
            x, y = st.columns([1, 8])
            with x:
                st.markdown(f"### {number}")
            with y:
                st.markdown(f"**{name}**")
                st.caption(description)
    st.divider()
    agent_question = st.text_area(
        "Define the business challenge",
        placeholder="Describe the business problem you want the agents to investigate...",
        height=110,
        key="agent_input"
    )
    agent_file = st.file_uploader(
        "Attach business dataset (optional)",
        type=["csv"],
        key="agent_dataset"
    )
    if st.button("🚀 Execute Agent Workflow", type="primary"):
        if not agent_question.strip():
            st.warning("Enter a business challenge.")
        elif not backend:
            st.error("Backend is offline.")
        else:
            try:
                with st.spinner("The four agents are collaborating..."):
                    if agent_file:
                        response = requests.post(
                            f"{API_URL}/api/agents/analyze-dataset",
                            data={"question": agent_question},
                            files={
                                "file": (
                                    agent_file.name,
                                    agent_file.getvalue(),
                                    "text/csv"
                                )
                            },
                            timeout=300
                        )
                    else:
                        response = requests.post(
                            f"{API_URL}/api/agents/run",
                            json={"question": agent_question},
                            timeout=300
                        )
                    response.raise_for_status()
                    st.session_state.agent_result = response.json()
                st.success("Agent workflow completed.")
            except requests.exceptions.RequestException as error:
                st.error(f"Agent execution failed: {error}")
    result = st.session_state.agent_result
    if result:
        workflow = result.get("agent_workflow", result)
        st.divider()
        x, y, z = st.columns(3)
        x.metric(
            "Agents Executed",
            workflow.get("agents_executed", 0)
        )
        y.metric(
            "Workflow Status",
            workflow.get("status", "Unknown").upper()
        )
        z.metric(
            "Dataset Attached",
            "Yes" if result.get("dataset_analysis") else "No"
        )
        st.markdown("### Execution Timeline")
        for log in workflow.get("execution_log", []):
            with st.expander(
                f"✓ {log.get('agent', 'Agent')} — {log.get('status', '')}"
            ):
                st.markdown(log.get("output", ""))
# =====================================================
# TAB 4 - EXECUTIVE REPORT
# =====================================================
with tab4:
    st.markdown(
        '<div class="section-heading">Executive Intelligence Reports</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Consolidated strategic findings and business recommendations.</div>',
        unsafe_allow_html=True
    )
    result = st.session_state.agent_result
    if result:
        workflow = result.get("agent_workflow", result)
        report = workflow.get("final_report")
        if report:
            st.success("Executive report generated.")
            with st.container(border=True):
                st.markdown("# NEXUS AI")
                st.caption("BUSINESS INTELLIGENCE REPORT")
                st.divider()
                st.markdown(report)
            st.markdown("### Export Your Executive Report")
            try:
                pdf_file = generate_pdf_report(report)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "⬇ Download PDF Report",
                        data=pdf_file,
                        file_name="NEXUS_AI_Executive_Report.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        "⬇ Download Markdown Report",
                        data=report,
                        file_name="NEXUS_AI_Executive_Report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            except Exception as error:
                st.error(f"Could not generate the PDF report: {error}")
                st.download_button(
                    "⬇ Download Markdown Report",
                    data=report,
                    file_name="NEXUS_AI_Executive_Report.md",
                    mime="text/markdown",
                    type="primary"
                )
        else:
            st.info("No final report available.")
    else:
        st.info(
            "Run the multi-agent workflow to generate "
            "your first executive report."
        )

# =====================================================
# TAB 5 - MONGODB REPORT HISTORY
# =====================================================

with tab5:
    st.markdown(
        '<div class="section-heading">Saved Executive Reports</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Browse, reopen, download and manage reports saved in MongoDB Atlas.</div>',
        unsafe_allow_html=True
    )

    refresh_col, info_col = st.columns([1, 4])
    with refresh_col:
        st.button(
            "↻ Refresh history",
            type="primary",
            key="refresh_report_history"
        )
    with info_col:
        st.caption("Reports are retrieved from the NEXUS AI FastAPI backend.")

    if not backend:
        st.error("Backend is offline. Start FastAPI to access saved reports.")
    else:
        try:
            history_response = requests.get(
                f"{API_URL}/api/reports",
                timeout=20
            )
            history_response.raise_for_status()
            history_payload = history_response.json()
            saved_reports = history_payload.get("reports", [])

            st.metric(
                "Saved executive reports",
                history_payload.get("count", len(saved_reports))
            )

            if not saved_reports:
                st.info(
                    "No saved reports yet. Run a workflow in Agent Studio "
                    "to generate and store your first report."
                )
            else:
                report_ids = [
                    item["id"] for item in saved_reports if item.get("id")
                ]
                report_lookup = {
                    item["id"]: item
                    for item in saved_reports if item.get("id")
                }

                selected_id = st.selectbox(
                    "Select a report to open",
                    options=report_ids,
                    index=(
                        report_ids.index(st.session_state.selected_saved_report_id)
                        if st.session_state.selected_saved_report_id in report_ids
                        else 0
                    ),
                    format_func=lambda report_id: (
                        f"{report_lookup[report_id].get('title') or 'Untitled report'}"
                        f"  ·  {report_lookup[report_id].get('created_at') or 'Date unavailable'}"
                    ),
                    key="saved_report_selector"
                )
                st.session_state.selected_saved_report_id = selected_id

                selected_summary = report_lookup[selected_id]
                meta1, meta2, meta3 = st.columns(3)
                meta1.metric(
                    "Workflow status",
                    str(selected_summary.get("workflow_status", "unknown")).upper()
                )
                meta2.metric(
                    "Dataset attached",
                    "Yes" if selected_summary.get("dataset_attached") else "No"
                )
                meta3.metric("Saved reports", len(saved_reports))

                detail_response = requests.get(
                    f"{API_URL}/api/reports/{selected_id}",
                    timeout=20
                )
                detail_response.raise_for_status()
                detail_payload = detail_response.json()
                report_document = detail_payload.get("report", {})
                report_text = report_document.get("report_content", "")

                st.divider()
                st.markdown(
                    "### " + (report_document.get("title") or "Executive Report")
                )
                if report_document.get("business_question"):
                    st.caption(
                        "Business question: "
                        + report_document["business_question"]
                    )
                if report_document.get("created_at"):
                    st.caption("Created: " + report_document["created_at"])

                if report_text:
                    with st.container(border=True):
                        st.markdown(report_text)

                    st.markdown("### Download saved report")
                    try:
                        pdf_data = generate_pdf_report(report_text)
                        download_pdf_col, download_md_col = st.columns(2)
                        safe_id = selected_id[-8:]

                        with download_pdf_col:
                            st.download_button(
                                "⬇ Download PDF",
                                data=pdf_data,
                                file_name=f"NEXUS_AI_Report_{safe_id}.pdf",
                                mime="application/pdf",
                                type="primary",
                                key=f"download_saved_pdf_{selected_id}",
                                use_container_width=True
                            )

                        with download_md_col:
                            st.download_button(
                                "⬇ Download Markdown",
                                data=report_text,
                                file_name=f"NEXUS_AI_Report_{safe_id}.md",
                                mime="text/markdown",
                                key=f"download_saved_md_{selected_id}",
                                use_container_width=True
                            )
                    except Exception as pdf_error:
                        st.error(f"Could not prepare the PDF: {pdf_error}")
                        st.download_button(
                            "⬇ Download Markdown",
                            data=report_text,
                            file_name=f"NEXUS_AI_Report_{selected_id[-8:]}.md",
                            mime="text/markdown",
                            key=f"download_saved_md_fallback_{selected_id}"
                        )
                else:
                    st.warning("This saved record does not contain report content.")

                st.divider()
                st.markdown("### Delete report")
                confirm_delete = st.checkbox(
                    "I understand this permanently deletes the selected report.",
                    key=f"confirm_delete_{selected_id}"
                )

                if st.button(
                    "Delete selected report",
                    disabled=not confirm_delete,
                    key=f"delete_saved_report_{selected_id}"
                ):
                    delete_response = requests.delete(
                        f"{API_URL}/api/reports/{selected_id}",
                        timeout=20
                    )
                    if delete_response.ok:
                        st.success("Report deleted successfully.")
                        st.session_state.selected_saved_report_id = None
                        st.rerun()
                    else:
                        st.error(
                            f"Could not delete report: "
                            f"{delete_response.status_code} - {delete_response.text}"
                        )

        except requests.exceptions.RequestException as error:
            st.error(f"Could not load saved reports from MongoDB: {error}")
        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Unexpected report-history response: {error}")


# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div class="footer">
    <strong>NEXUS AI</strong><br>
    Autonomous Multi-Agent Business Intelligence Platform<br>
    Python · FastAPI · Groq · Pandas · Plotly · Streamlit
</div>
""", unsafe_allow_html=True)
