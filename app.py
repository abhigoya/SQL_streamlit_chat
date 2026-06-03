import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
import logging
import pathlib
import os
import re
import io
import pymysql
import pandas as pd
from dotenv import load_dotenv
import datetime as dt
from datetime import datetime
import dashboard

load_dotenv(override=True)

# --- Logging setup ---
log_dir = pathlib.Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
log.info("Streamlit app started")

st.set_page_config(page_title="Conf AI Dashboard", page_icon="🗄️", layout="wide")

# --- API Key gate ---
# Always ask the user for their key on a fresh session; never auto-fill from .env
if "qgenie_api_key" not in st.session_state:
    st.session_state.qgenie_api_key = ""
if "api_key_error" not in st.session_state:
    st.session_state.api_key_error = ""

if not st.session_state.qgenie_api_key:
    st.markdown("Please enter your **QGenie API Key** to get started.")
    if st.session_state.api_key_error:
        st.error(st.session_state.api_key_error)
    with st.form("api_key_form"):
        entered_key = st.text_input("QGenie API Key", type="password", placeholder="Enter your API key...")
        st.markdown('<p style="color:#4A90D9;">Get your QGENIE_API_KEY from: <a href="https://qpilot.qualcomm.com/api_key" style="color:#F4A261; font-weight:bold;">https://qpilot.qualcomm.com/api_key</a></p>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Continue")
        if submitted:
            if entered_key.strip():
                st.session_state.qgenie_api_key = entered_key.strip()
                st.session_state.api_key_error = ""
                os.environ["QGENIE_API_KEY"] = entered_key.strip()
                st.rerun()
            else:
                st.error("API key cannot be empty.")
    st.stop()

# Key is confirmed — make it available in os.environ for downstream use
os.environ["QGENIE_API_KEY"] = st.session_state.qgenie_api_key

# Import agent only after the key is in os.environ so QGenieChat picks up the real key
from SQL_langrapgh import agent

st.title("Conf AI Dashboard")
st.caption("Ask questions about the realtimedata database.")

# --- Filter helpers ---
@st.cache_data(show_spinner=False)
def fetch_distinct(column: str) -> list:
    """Return sorted distinct non-null values for a column in realtimedata."""
    try:
        con = pymysql.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "3306")),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        cur = con.cursor()
        cur.execute(f"SELECT DISTINCT {column} FROM realtimedata WHERE {column} IS NOT NULL ORDER BY {column}")
        values = [row[0] for row in cur.fetchall()]
        con.close()
        return values
    except Exception as e:
        log.warning(f"Could not fetch distinct {column}: {e}")
        return []


def _db_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


@st.cache_data(show_spinner=False)
def fetch_setup_combined() -> list:
    """Return 'LAB1234: KEYS - TMO 5G_Protocol' style entries."""
    try:
        con = _db_conn()
        cur = con.cursor()
        cur.execute("""
            SELECT DISTINCT CONCAT(sl.setupname, ': ', sl.setup, ' - ', sl.operator, ' ', sl.area) AS display
            FROM realtimedata r
            LEFT JOIN setup_labels sl ON r.setupname = sl.setupname
            WHERE sl.setup IS NOT NULL AND sl.operator IS NOT NULL AND sl.area IS NOT NULL
            ORDER BY display
        """)
        values = [row[0] for row in cur.fetchall()]
        con.close()
        return values
    except Exception as e:
        log.warning(f"Could not fetch setup combined labels: {e}")
        return []


def resolve_combined_setups(selected: tuple) -> tuple:
    """Extract setupnames from 'LAB1234: KEYS - TMO 5G_Protocol' labels."""
    if not selected:
        return ()
    return tuple(v.split(": ")[0] for v in selected)

# Persist conversation history and reasoning traces across reruns
if "history" not in st.session_state:
    st.session_state.history: list[BaseMessage] = []
if "traces" not in st.session_state:
    st.session_state.traces: list[list[dict]] = []  # one trace list per turn

# Sidebar: filters + clear history button
with st.sidebar:
    st.header("Filters")
    _ALLOWED_CARRIERS = {"TMO", "ATT", "AT&T", "GCF", "VZW"}
    carrier_options = [c for c in fetch_distinct("carrier") if c in _ALLOWED_CARRIERS]
    selected_carrier = st.multiselect("Carrier", carrier_options, placeholder="All carriers")
    selected_setup = st.multiselect("Setup Name", fetch_setup_combined(), placeholder="All setups")
    st.caption("Start Time")
    s_col1, s_col2 = st.columns(2)
    start_date = s_col1.date_input("Date", value=None, key="start_date", label_visibility="collapsed")
    start_time = s_col2.time_input("Time", value=dt.time(0, 0, 0), key="start_time", label_visibility="collapsed")
    st.caption("End Time")
    e_col1, e_col2 = st.columns(2)
    end_date = e_col1.date_input("Date", value=None, key="end_date", label_visibility="collapsed")
    end_time = e_col2.time_input("Time", value=dt.time(23, 59, 59), key="end_time", label_visibility="collapsed")
    selected_starttime = dt.datetime.combine(start_date, start_time) if start_date else None
    selected_endtime = dt.datetime.combine(end_date, end_time) if end_date else None
    st.divider()
    st.header("Options")
    if st.button("Clear conversation"):
        log.info("Conversation cleared by user")
        st.session_state.history = []
        st.session_state.traces = []
        st.rerun()
    st.divider()
    st.caption(f"Log file: `{log_file}`")


_TABLE_RE = re.compile(
    r'(\|.+\|[ \t]*\n\|[-| :]+\|[ \t]*\n(?:\|.+\|[ \t]*\n?)+)',
    re.MULTILINE,
)


def _parse_md_table(text: str) -> pd.DataFrame:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = [
        [c.strip() for c in line.strip("|").split("|")]
        for line in lines[2:]  # skip header + separator
        if line.startswith("|")
    ]
    return pd.DataFrame(rows, columns=headers)


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buf.getvalue()


def render_response(text: str, render_key: str = ""):
    """Render LLM output — detected markdown tables become st.dataframe blocks."""
    parts = _TABLE_RE.split(text)
    df_idx = 0
    for part in parts:
        if not part.strip():
            continue
        if _TABLE_RE.match(part.strip()):
            try:
                df = _parse_md_table(part)
                st.dataframe(df, use_container_width=True, hide_index=True)
                col1, col2 = st.columns([2, 5])
                with col1:
                    try:
                        st.download_button(
                            "⬇ Download as Excel",
                            data=df_to_excel_bytes(df),
                            file_name="results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_{render_key}_{df_idx}",
                            use_container_width=True,
                        )
                    except Exception:
                        pass
                if len(df) >= 20:
                    with col2:
                        st.info(
                            "Showing 20 rows — results may be truncated. "
                            "Ask: *'export all results to Excel'* for the complete dataset."
                        )
                df_idx += 1
            except Exception:
                st.markdown(part)
        else:
            st.markdown(part)


def render_reasoning(trace: list[dict]):
    """Render a collapsible reasoning block for one turn."""
    with st.expander("Reasoning / Steps", expanded=False):
        for step in trace:
            if step["type"] == "tool_call":
                st.markdown(f"**Tool:** `{step['name']}`")
                st.code(step["args"], language="sql" if "query" in step["name"] else "text")
            elif step["type"] == "tool_result":
                st.markdown(f"**Result from** `{step['name']}`")
                st.code(step["content"], language="text")


def extract_trace(messages: list[BaseMessage]) -> list[dict]:
    """Pull tool calls and their results out of a message list."""
    trace = []
    tool_call_names: dict[str, str] = {}  # id -> tool name

    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                tool_call_names[tc["id"]] = tc["name"]
                args_str = tc["args"].get("query") or tc["args"].get("table_names") or str(tc["args"])
                trace.append({"type": "tool_call", "name": tc["name"], "args": args_str})
        elif isinstance(m, ToolMessage):
            name = tool_call_names.get(m.tool_call_id, "tool")
            trace.append({"type": "tool_result", "name": name, "content": m.content})
    return trace


_NODE_LABELS = {
    "classify_query":  "Classifying question type",
    "list_tables":     "Listing available tables",
    "call_get_schema": "Fetching table schemas",
    "get_schema":      "Reading schema",
    "generate_query":  "Generating query",
    "check_query":     "Validating SQL",
    "run_query":       "Running query",
    "run_jira":        "Fetching JIRA data",
}


def _render_step(node: str, node_state: dict, label: str) -> None:
    """Write one agent step inside the active st.status container."""
    msgs = node_state.get("messages", [])
    last_msg = msgs[-1] if msgs else None

    if node == "classify_query":
        qt = node_state.get("query_type", "")
        st.write("**" + label + "**" + (f"  →  `{qt}`" if qt else ""))
        return

    if last_msg is not None and getattr(last_msg, "tool_calls", None):
        tc = last_msg.tool_calls[0]
        st.write(f"**{label}**")
        if tc["name"] == "sql_db_query":
            sql = tc["args"].get("query", "")
            if sql:
                st.code(sql, language="sql")
        return

    st.write(f"**{label}**")


if "dashboard_loaded" not in st.session_state:
    st.session_state.dashboard_loaded = False

tab_chat, tab_dashboard = st.tabs(["Chat", "Dashboard"])

with tab_dashboard:
    if not st.session_state.dashboard_loaded:
        st.info("Click the button below to load the dashboard.")
        if st.button("Load Dashboard"):
            st.session_state.dashboard_loaded = True
            st.rerun()
    else:
        if selected_setup:
            resolved_setups = resolve_combined_setups(tuple(selected_setup))
            if not resolved_setups:
                st.info("No setups match the selected Setup filters.")
                resolved_setups = ('__no_match__',)
        else:
            resolved_setups = ()
        dashboard.render(tuple(selected_carrier), resolved_setups, selected_starttime, selected_endtime)

with tab_chat:
    # Render existing chat messages with their reasoning traces
    turn_index = 0
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls and not msg.content.startswith("Available tables:") and "<|" not in msg.content:
            with st.chat_message("assistant"):
                if turn_index < len(st.session_state.traces) and st.session_state.traces[turn_index]:
                    render_reasoning(st.session_state.traces[turn_index])
                render_response(msg.content, render_key=f"t{turn_index}")
                turn_index += 1

    # User input
    if user_query := st.chat_input("Ask a question about the database..."):
        # Build filter context string to prepend to the query
        filter_parts = []
        if selected_carrier:
            vals = ", ".join(f"'{v}'" for v in selected_carrier)
            filter_parts.append(f"carrier IN ({vals})")
        if selected_setup:
            resolved = resolve_combined_setups(tuple(selected_setup))
            if resolved:
                vals = ", ".join(f"'{v}'" for v in resolved)
                filter_parts.append(f"setupname IN ({vals})")
            else:
                filter_parts.append("setupname IN ('__no_matching_setup__')")
        if selected_starttime:
            filter_parts.append(f"starttime >= '{selected_starttime.strftime('%Y-%m-%d:%H:%M:%S')}'")
        if selected_endtime:
            filter_parts.append(f"endtime <= '{selected_endtime.strftime('%Y-%m-%d:%H:%M:%S')}'")

        if filter_parts:
            user_query = f"[Filters: {', '.join(filter_parts)}] {user_query}"
        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.history.append(HumanMessage(content=user_query))

        with st.chat_message("assistant"):
            final_answer = "I was unable to generate a response. Please try rephrasing your question."
            latest_state = None

            with st.status("Analyzing your question...", expanded=True) as status:
                try:
                    log.info(f"USER: {user_query}")
                    for event_mode, data in agent.stream(
                        {"messages": st.session_state.history, "query_type": ""},
                        stream_mode=["updates", "values"],
                        config={"recursion_limit": 50},
                    ):
                        if event_mode == "updates":
                            node = next(iter(data))
                            node_state = data[node]
                            label = _NODE_LABELS.get(node, node.replace("_", " ").title())
                            status.update(label=label)
                            _render_step(node, node_state, label)
                        elif event_mode == "values":
                            latest_state = data

                    status.update(label="Done", state="complete", expanded=False)

                except Exception as e:
                    log.exception(f"ERROR during agent stream: {e}")
                    status.update(label="Error", state="error", expanded=False)
                    if "401" in str(e) or "UNAUTHORIZED" in str(e):
                        log.warning("Invalid API key — resetting key gate")
                        st.session_state.qgenie_api_key = ""
                        st.session_state.api_key_error = "Invalid or expired API key. Please enter a valid key."
                        import sys; sys.modules.pop("SQL_langrapgh", None)
                        st.rerun()
                    final_answer = f"An error occurred: {e}"
                    st.session_state.traces.append([])

            if latest_state is not None:
                response_messages = latest_state["messages"]

                for m in response_messages:
                    if isinstance(m, AIMessage):
                        if m.tool_calls:
                            log.debug(f"TOOL CALLS: {m.tool_calls}")
                        if m.content:
                            log.info(f"AGENT: {m.content}")
                    elif hasattr(m, "content"):
                        log.debug(f"{type(m).__name__}: {m.content}")

                final_answer = next(
                    (
                        m.content
                        for m in reversed(response_messages)
                        if isinstance(m, AIMessage) and m.content and not m.tool_calls
                        and not m.content.startswith("Available tables:")
                        and "<|" not in m.content
                    ),
                    "I was unable to generate a response. Please try rephrasing your question.",
                )
                log.info(f"FINAL ANSWER: {final_answer}")

                new_messages = response_messages[len(st.session_state.history):]
                trace = extract_trace(new_messages)
                st.session_state.traces.append(trace)
                st.session_state.history = response_messages

            render_response(final_answer, render_key="current")

        st.rerun()
