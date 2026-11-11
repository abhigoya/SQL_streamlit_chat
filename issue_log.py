import os
from datetime import date, time

import pymysql
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

ISSUE_TYPES = ["Setup", "UE", "TE", "Other"]


def _get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


@st.cache_data(show_spinner=False, ttl=300)
def fetch_setup_names() -> list:
    con = _get_connection()
    try:
        with con.cursor() as cur:
            cur.execute("SELECT DISTINCT setupname FROM realtimedata WHERE setupname IS NOT NULL ORDER BY setupname")
            return [row[0] for row in cur.fetchall()]
    finally:
        con.close()


def insert_issue(
    setupname: str,
    issue_type: str,
    issue_date: date,
    start_time: time,
    end_time: time,
    description: str,
    ticket_id: str | None = None,
    end_date: date | None = None,
) -> None:
    con = _get_connection()
    try:
        with con.cursor() as cur:
            cur.execute(
                "INSERT INTO overnight_issues (setupname, issue_type, issue_date, start_time, end_time, end_date, description, ticket_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    setupname,
                    issue_type,
                    issue_date.isoformat(),
                    start_time.isoformat(),
                    end_time.isoformat(),
                    (end_date or issue_date).isoformat(),
                    description,
                    ticket_id,
                ),
            )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def render_issue_log_tab():
    st.subheader("Log an Overnight Issue")

    with st.form("issue_log_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            setupname = st.selectbox("Setup Name", options=fetch_setup_names())
        with col2:
            issue_type = st.selectbox("Issue Type", options=ISSUE_TYPES)
        with col3:
            ticket_id = st.text_input("Ticket ID", placeholder="Optional — UE/TE ticket ID")

        col4, col5, _spacer1 = st.columns([1, 1, 2])
        with col4:
            issue_date = st.date_input("Start Date", value=date.today())
        with col5:
            start_time = st.time_input("Start Time", value=time(0, 0, 0))

        col6, col7, _spacer2 = st.columns([1, 1, 2])
        with col6:
            end_date = st.date_input("End Date", value=date.today())
        with col7:
            end_time = st.time_input("End Time", value=time(23, 59, 59))

        description = st.text_area("Description", placeholder="Describe the issue (e.g. licenses missing, UE not responding)...")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if not description.strip():
            st.error("Description cannot be empty.")
        else:
            try:
                insert_issue(setupname, issue_type, issue_date, start_time, end_time, description.strip(), ticket_id.strip() or None, end_date)
                st.success(f"Issue logged for {setupname} on {issue_date}.")
            except Exception as e:
                st.error(f"Failed to save issue: {e}")
