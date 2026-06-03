import io
import os
import pandas as pd
import pymysql
import streamlit as st
from dotenv import load_dotenv
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, JsCode

load_dotenv(override=True)


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
            MAiLAF
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


def _update_rows(test_ids: list, ue_build: str, test_result: str) -> int:
    """Update ue_build and/or testresult for the given testid values.

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

    export_cols = ["carrier","setupname", "starttime", "testid", "testresult", "Execmode", "duration", "ue_build", "TEBuild", "TE_file_path", "UE_log_path", "Reason", "MAiLAF"]
    all_tests_export = all_tests[export_cols].rename(columns={
        "setupname": "Setup",
        "testid": "Test ID",
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
        "MAiLAF": "MAiLAF",
    })

    # --- Excel-like set filters ---
    f1, f2, f3 = st.columns(3)
    f4, f5, f6 = st.columns(3)
    sel_setupname = f1.multiselect("Setup Name", sorted(all_tests_export["Setup"].dropna().unique()))
    sel_test_id   = f2.multiselect("Test ID",    sorted(all_tests_export["Test ID"].dropna().unique()))
    sel_result    = f3.multiselect("Result",     sorted(all_tests_export["Result"].dropna().unique()))
    sel_carrier   = f4.multiselect("Carrier",    sorted(all_tests_export["Carrier"].dropna().unique()))
    sel_ue_build  = f5.multiselect("UE Build",   sorted(all_tests_export["UE Build"].dropna().unique()))
    sel_te_build  = f6.multiselect("TE Build",   sorted(all_tests_export["TE Build"].dropna().unique()))

    filtered_export = all_tests_export.copy()
    if sel_setupname:
        filtered_export = filtered_export[filtered_export["Setup"].isin(sel_setupname)]
    if sel_test_id:
        filtered_export = filtered_export[filtered_export["Test ID"].isin(sel_test_id)]
    if sel_result:
        filtered_export = filtered_export[filtered_export["Result"].isin(sel_result)]
    if sel_carrier:
        filtered_export = filtered_export[filtered_export["Carrier"].isin(sel_carrier)]
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
            submitted = st.form_submit_button("Update Selected")

        if submitted:
            if not new_ue_build.strip() and not new_test_result:
                st.warning("No fields to update — enter a UE Build or select a Test Result.")
            else:
                test_ids = selected_rows["Test ID"].tolist()
                try:
                    rows_updated = _update_rows(test_ids, new_ue_build, new_test_result)
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
