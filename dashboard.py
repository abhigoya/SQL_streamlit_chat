import io
import os
import pandas as pd
import pymysql
import streamlit as st
from dotenv import load_dotenv
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, JsCode

load_dotenv(override=True)

JIRA_BASE_URL = os.getenv("JIRA_BROWSE_URL", os.getenv("JIRA_URL", "")).rstrip("/")
ORBIT_BASE_URL = os.getenv("ORBIT_URL", "https://orbit").rstrip("/")


def _get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


@st.cache_data(show_spinner=False, ttl=300)
def _load_data(carrier: tuple, setupname: tuple, starttime, endtime) -> pd.DataFrame:
    """Query realtimedata with the active sidebar filters applied."""
    conditions = ["1=1"]
    params = []

    if carrier:
        placeholders = ", ".join(["%s"] * len(carrier))
        conditions.append(f"carrier IN ({placeholders})")
        params.extend(carrier)
    if setupname:
        placeholders = ", ".join(["%s"] * len(setupname))
        conditions.append(f"setupname IN ({placeholders})")
        params.extend(setupname)
    if starttime:
        conditions.append("starttime >= %s")
        params.append(starttime.strftime("%Y-%m-%d:%H:%M:%S"))
    if endtime:
        conditions.append("endtime <= %s")
        params.append(endtime.strftime("%Y-%m-%d:%H:%M:%S"))

    where = " AND ".join(conditions)
    query = f"""
        SELECT
            testid,
            carrier,
            setupname,
            starttime,
            endtime,
            duration,
            testresult,
            Execmode,
            ue_build,
            TEBuild,
            TE_file_path,
            UE_log_path,
            Reason,
            NeedsRCA,
            MAiLAF,
            Claude,
            comments
        FROM realtimedata
        WHERE {where}
    """
    con = _get_connection()
    df = pd.read_sql_query(query, con, params=params)
    con.close()

    # Parse starttime to date for daily grouping
    df["date"] = pd.to_datetime(
        df["starttime"], format="%Y-%m-%d:%H:%M:%S", errors="coerce"
    ).dt.date

    return df


@st.cache_data(show_spinner=False, ttl=300)
def _load_jira_matches(testids: tuple) -> dict:
    """Batch-match realtimedata testids against jira_tickets.test_id via suffix LIKE.

    jira_tickets.test_id is stored in an inconsistent format (e.g. CONF+TMO+<testid>,
    or no prefix at all), so matches use a suffix LIKE rather than an exact match —
    same convention used by the chat agent (SQL_langrapgh.py).

    Uses a temporary table rather than a UNION ALL values-list: with thousands of
    distinct testids, a long UNION ALL chain overflows MySQL's thread stack.

    Also returns jira_tickets.cr (Orbit CR number synced from JIRA's Orbit-CR custom
    field) alongside the ticket key — this only covers CRs that went through a JIRA
    ticket; CRs raised directly in Orbit have no JIRA row and won't appear here.
    """
    if not testids:
        return {}
    con = _get_connection()
    try:
        with con.cursor() as cur:
            cur.execute("SET SESSION group_concat_max_len = 10000")
            cur.execute("CREATE TEMPORARY TABLE tmp_testids (testid VARCHAR(255) PRIMARY KEY)")
            cur.executemany(
                "INSERT IGNORE INTO tmp_testids (testid) VALUES (%s)",
                [(t,) for t in testids],
            )
            cur.execute("""
                SELECT v.testid AS rd_testid,
                       GROUP_CONCAT(DISTINCT jt.ticket_key ORDER BY jt.ticket_key SEPARATOR ',') AS ticket_keys,
                       GROUP_CONCAT(DISTINCT jt.cr ORDER BY jt.cr SEPARATOR ',') AS crs
                FROM tmp_testids v
                JOIN jira_tickets jt ON jt.test_id LIKE CONCAT('%%', v.testid)
                GROUP BY v.testid
            """)
            rows = cur.fetchall()
    finally:
        con.close()
    return {row[0]: {"ticket_keys": row[1], "crs": row[2]} for row in rows}


def _update_rows(test_ids: list, ue_build: str, test_result: str, comments: str) -> int:
    """Update ue_build, testresult, and/or comments for the given testid values.

    Only non-empty fields are included in the SET clause.
    Returns the number of rows actually updated.
    """
    set_parts, params = [], []
    if ue_build.strip():
        set_parts.append("ue_build = %s")
        params.append(ue_build.strip())
    if test_result:
        set_parts.append("testresult = %s")
        params.append(test_result)
    if comments.strip():
        set_parts.append("comments = %s")
        params.append(comments.strip())
    if not set_parts:
        return 0
    set_clause = ", ".join(set_parts)
    placeholders = ", ".join(["%s"] * len(test_ids))
    sql = f"UPDATE realtimedata SET {set_clause} WHERE testid IN ({placeholders})"
    params.extend(test_ids)
    con = _get_connection()
    try:
        with con.cursor() as cur:
            cur.execute(sql, params)
        con.commit()
        return cur.rowcount
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def render(carrier: tuple, setupname: tuple, starttime, endtime):
    """Render the dashboard tab content."""
    try:
        df = _load_data(carrier, setupname, starttime, endtime)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if df.empty:
        st.info("No data found for the selected filters.")
        return

    total = len(df)
    pass_count = (df["testresult"] == "PASS").sum()
    fail_count = (df["testresult"] == "FAIL").sum()
    inconclusive_count = (df["testresult"] == "INCONCLUSIVE").sum()
    pass_rate = round(pass_count / total * 100, 1) if total else 0
    total_duration_sec = pd.to_timedelta(df["duration"].astype(str).str.strip(), errors="coerce").dt.total_seconds().fillna(0).sum()
    total_duration_fmt = f"{int(total_duration_sec // 3600):02d}:{int((total_duration_sec % 3600) // 60):02d}:{int(total_duration_sec % 60):02d}"

    # --- KPI row ---
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Executed", total)
    k2.metric("Pass", pass_count)
    k3.metric("Fail", fail_count)
    k4.metric("Inconclusive", inconclusive_count)
    k5.metric("Pass Rate", f"{pass_rate}%")
    k6.metric("Total Duration", total_duration_fmt)

    st.divider()

    # --- All test cases table ---
    st.subheader("Test Cases")
    all_tests = df.sort_values("starttime", ascending=False).copy()

    jira_matches = _load_jira_matches(tuple(sorted(all_tests["testid"].dropna().unique())))
    all_tests["jira_id"] = all_tests["testid"].map(lambda t: jira_matches.get(t, {}).get("ticket_keys"))
    all_tests["cr"] = all_tests["testid"].map(lambda t: jira_matches.get(t, {}).get("crs"))
    all_tests.loc[all_tests["testresult"] == "PASS", ["jira_id", "cr"]] = None

    export_cols = ["carrier","setupname", "starttime", "testid", "testresult", "Execmode", "duration", "ue_build", "TEBuild", "TE_file_path", "UE_log_path", "Reason", "NeedsRCA", "MAiLAF", "Claude", "jira_id", "cr", "comments"]
    all_tests_export = all_tests[export_cols].rename(columns={
        "setupname": "Setup",
        "testid": "Test ID",
        "jira_id": "JIRA ID",
        "cr": "CR",
        "carrier": "Carrier",
        "testresult": "Result",
        "Execmode": "Exec Mode",
        "starttime": "Start Time",
        "duration": "Duration",
        "ue_build": "UE Build",
        "TEBuild": "TE Build",
        "TE_file_path": "TE Log Path",
        "UE_log_path": "UE Log Path",
        "Reason": "TE Fail Reason",
        "NeedsRCA": "AutoD",
        "MAiLAF": "MAiLAF",
        "Claude": "Claude",
        "comments": "Manual Comments",
    })

    # --- Excel-like set filters ---
    f1, f2, f3 = st.columns(3)
    sel_result   = f1.multiselect("Result",   sorted(all_tests_export["Result"].dropna().unique()))
    sel_ue_build = f2.multiselect("UE Build", sorted(all_tests_export["UE Build"].dropna().unique()))
    sel_te_build = f3.multiselect("TE Build", sorted(all_tests_export["TE Build"].dropna().unique()))

    filtered_export = all_tests_export.copy()
    if sel_result:
        filtered_export = filtered_export[filtered_export["Result"].isin(sel_result)]
    if sel_ue_build:
        filtered_export = filtered_export[filtered_export["UE Build"].isin(sel_ue_build)]
    if sel_te_build:
        filtered_export = filtered_export[filtered_export["TE Build"].isin(sel_te_build)]

    gb = GridOptionsBuilder.from_dataframe(filtered_export)
    gb.configure_default_column(filter=True, sortable=True, resizable=True, floatingFilter=False)
    gb.configure_grid_options(domLayout="normal", enableCellTextSelection=True, ensureDomOrder=True)
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
        header_checkbox=True,
        header_checkbox_filtered_only=True,
    )

    _link_renderer = JsCode("""
        class LinkRenderer {
            init(params) {
                this.eGui = document.createElement('span');
                if (params.value && params.value.trim() !== '') {
                    var a = document.createElement('a');
                    a.href = params.value;
                    a.innerText = 'Open';
                    a.target = '_blank';
                    a.style.color = '#4da6ff';
                    a.style.textDecoration = 'underline';
                    this.eGui.appendChild(a);
                }
            }
            getGui() { return this.eGui; }
        }
    """)
    gb.configure_column("TE Log Path", cellRenderer=_link_renderer)
    gb.configure_column("UE Log Path", cellRenderer=_link_renderer)
    gb.configure_column("MAiLAF", cellRenderer=_link_renderer)
    gb.configure_column("Claude", cellRenderer=_link_renderer)

    _jira_link_renderer = JsCode(f"""
        class JiraLinkRenderer {{
            init(params) {{
                this.eGui = document.createElement('span');
                if (params.value && params.value.trim() !== '') {{
                    var keys = params.value.split(',').map(s => s.trim()).filter(Boolean);
                    keys.forEach((key, idx) => {{
                        var a = document.createElement('a');
                        a.href = '{JIRA_BASE_URL}/browse/' + key;
                        a.innerText = key;
                        a.target = '_blank';
                        a.style.color = '#4da6ff';
                        a.style.textDecoration = 'underline';
                        this.eGui.appendChild(a);
                        if (idx < keys.length - 1) this.eGui.appendChild(document.createTextNode(', '));
                    }});
                }}
            }}
            getGui() {{ return this.eGui; }}
        }}
    """)
    gb.configure_column("JIRA ID", cellRenderer=_jira_link_renderer)

    _cr_link_renderer = JsCode(f"""
        class CrLinkRenderer {{
            init(params) {{
                this.eGui = document.createElement('span');
                if (params.value && params.value.trim() !== '') {{
                    var crs = params.value.split(',').map(s => s.trim()).filter(Boolean);
                    crs.forEach((cr, idx) => {{
                        var a = document.createElement('a');
                        a.href = '{ORBIT_BASE_URL}/CR/' + cr;
                        a.innerText = cr;
                        a.target = '_blank';
                        a.style.color = '#4da6ff';
                        a.style.textDecoration = 'underline';
                        this.eGui.appendChild(a);
                        if (idx < crs.length - 1) this.eGui.appendChild(document.createTextNode(', '));
                    }});
                }}
            }}
            getGui() {{ return this.eGui; }}
        }}
    """)
    gb.configure_column("CR", cellRenderer=_cr_link_renderer)

    grid_options = gb.build()

    grid_response = AgGrid(
        filtered_export,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_on=["selectionChanged", "filterChanged"],
        use_container_width=True,
        height=600,
        key="test_cases_grid",
        allow_unsafe_jscode=True,
    )

    filtered_count = len(grid_response.data)
    total_count = len(all_tests_export)
    if filtered_count < total_count:
        st.caption(f"Showing {filtered_count} of {total_count} test case(s)")
    else:
        st.caption(f"{total_count} test case(s)")

    # --- Inline update for selected rows ---
    selected_rows = grid_response.selected_rows  # pd.DataFrame or None in st-aggrid 1.2.1

    if selected_rows is not None and not selected_rows.empty:
        n = len(selected_rows)
        st.markdown(f"**{n} row(s) selected** — update fields below and submit.")

        with st.form(key="update_selected_form", clear_on_submit=True):
            new_ue_build = st.text_input("New UE Build", value="", placeholder="Leave blank to skip")
            new_test_result = st.selectbox(
                "New Test Result",
                options=["", "PASS", "FAIL", "INCONCLUSIVE"],
                index=0,
                format_func=lambda x: "(no change)" if x == "" else x,
            )
            new_comments = st.text_area("Comments", value="", placeholder="Leave blank to skip")
            submitted = st.form_submit_button("Update Selected")

        if submitted:
            if not new_ue_build.strip() and not new_test_result and not new_comments.strip():
                st.warning("No fields to update — enter a UE Build, select a Test Result, or add a Comment.")
            else:
                test_ids = selected_rows["Test ID"].tolist()
                try:
                    rows_updated = _update_rows(test_ids, new_ue_build, new_test_result, new_comments)
                    if rows_updated > 0:
                        st.success(f"Updated {rows_updated} row(s) successfully.")
                    else:
                        st.info("No rows were modified (values may already match).")
                    _load_data.clear()  # invalidate cache so grid reflects new data
                    st.rerun()
                except Exception as e:
                    st.error(f"Database update failed: {e}")

    # --- Export button ---
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        all_tests_export.to_excel(writer, index=False, sheet_name="Test Cases")
    excel_data = excel_buffer.getvalue()

    st.download_button(
        label="Export Excel",
        data=excel_data,
        file_name="test_cases.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
