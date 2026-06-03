# Conf AI Dashboard

A LangGraph-powered AI agent that lets users query a MySQL test-results database and JIRA tickets through natural language, exposed as a Streamlit web app.

## Features

- **Natural language SQL** — ask questions about test results, pass/fail rates, and execution history; the agent generates and runs the SQL for you
- **JIRA integration** — look up tickets by test ID, ticket key (`MSTCONF-XXXXX`), or keyword; queries the local cache or falls back to the live JIRA API
- **Combined queries** — the agent detects when a question requires both a SQL lookup and a JIRA lookup and handles both in one response
- **Interactive dashboard** — filterable ag-Grid with KPI cards, row-level editing, clickable log-path links, and Excel export
- **Reasoning trace** — every agent response shows the tool-call trace in a collapsible "Reasoning" expander

## Prerequisites

| Tool | Version |
|------|---------|
| Python | >= 3.10 |
| MySQL | >= 8.0 (remote or local) |
| JIRA instance | Accessible via HTTP |

## Installation & Setup

```bash
# 1. Clone / unzip the project
cd SQL_backup

# 2. Install dependencies (no manifest included — install manually)
pip install streamlit langchain-core langchain-community langgraph \
            pymysql pandas st-aggrid python-dotenv atlassian-python-api \
            openpyxl qgenie qgenie-sdk-tools

# 3. Create a .env file in the project root
DB_HOST=<mysql-host>
DB_PORT=3306
DB_NAME=<database-name>
DB_USER=<username>
DB_PASSWORD=<password>
JIRA_URL=https://<your-jira-instance>
JIRA_USERNAME=<jira-email>
JIRA_PASSWORD=<jira-api-token>

# 4. (One-time) Sync JIRA tickets into the local cache table
python jira_sync.py --full-sync

# 5. Launch the web app
streamlit run app.py
```

When the app opens, enter your **QGenie API key** in the prompt. The key is stored only in the browser session and is never written to disk.

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework — chat interface, dashboard, sidebar filters |
| `langgraph` | State-machine agent that routes and orchestrates tool calls |
| `langchain-core` | LLM message types and tool abstractions |
| `langchain-community` | `SQLDatabase` toolkit for schema inspection and query execution |
| `qgenie` / `qgenie.integrations.langchain` | Claude model wrapper (Anthropic) used as the agent LLM |
| `qgenie-sdk-tools` | Pre-built JIRA tools (`read_jira_ticket`, `search_jira_issues`, `custom_jira_tool`) |
| `pymysql` | MySQL driver (used via SQLAlchemy URI) |
| `pandas` | DataFrame manipulation for KPI calculations and inline table rendering |
| `st-aggrid` | Interactive ag-Grid component for the Dashboard tab |
| `atlassian-python-api` | JIRA REST API client used by `jira_sync.py` |
| `python-dotenv` | Loads `.env` into `os.environ` at startup |
| `openpyxl` | Excel export from the dashboard (used via `pandas.to_excel`) |

## Project Structure

```
SQL_backup/
├── app.py              # Streamlit entry point — auth gate, chat UI, dashboard tab
├── SQL_langrapgh.py    # LangGraph agent — query classification, SQL/JIRA routing
├── dashboard.py        # Dashboard tab — ag-Grid, KPIs, row editing, Excel export
├── jira_sync.py        # CLI script — syncs JIRA tickets into MySQL cache table
└── .env                # (not committed) database and JIRA credentials
```

## File Reference

**[app.py](app.py)** — Streamlit application entry point. Renders the API-key gate on first load, a sidebar with `carrier` / `setupname` / time-range filters, and two tabs: **Chat** (calls `SQL_langrapgh.agent`) and **Dashboard** (delegates to `dashboard.render()`). Prepends active filter context to every user query before sending it to the agent. Parses markdown tables in responses and renders them as interactive DataFrames. Shows tool-call traces in a collapsible "Reasoning" expander.

**[SQL_langrapgh.py](SQL_langrapgh.py)** — The LangGraph agent. Classifies each query as `sql`, `jira`, or `combined`, then routes it through a state machine:

```
START → classify_query → list_tables → call_get_schema → get_schema
      → generate_query → check_query → run_query (SQL path)
                       → run_jira              (JIRA path)
```

Connects to MySQL via SQLAlchemy, uses the QGenie Claude model (`claude-4-5-sonnet`), and enforces a 20-row default result limit. Handles the non-standard datetime format `YYYY-MM-DD:HH:MM:SS` and LIKE-based joins for test IDs that carry prefixes in the JIRA table.

**[dashboard.py](dashboard.py)** — Self-contained dashboard module imported by `app.py`. Queries the `realtimedata` table with sidebar filters, computes KPIs (pass rate, total duration, result breakdown), and displays results in an ag-Grid with multi-select column filters, clickable log-path links, and row-level editing (UE Build / Test Result fields). Caches data for 300 seconds. Exports filtered data to Excel.

**[jira_sync.py](jira_sync.py)** — Standalone CLI script for keeping `jira_tickets` in sync with the live JIRA instance. Three modes: `--load-csv` (seed from a CSV dump), `--full-sync` (fetch all tickets from the past 5 years), and default incremental (fetch only tickets updated since the last run). Creates the `jira_tickets` table if it does not exist and upserts rows via `ON DUPLICATE KEY UPDATE`.

## Architecture & Data Flow

```
Browser
  │
  ▼
app.py  ──────────────────────────────────────────────────────────┐
  │  Sidebar filters                                              │
  │  Chat input                                                   │
  │  [Filters: carrier IN (...)] + user question                 │
  │                                                              │
  ├── dashboard.py ──── SELECT * FROM realtimedata WHERE ...    │
  │        │                │                                    │
  │        ▼                ▼                                    │
  │      KPIs         ag-Grid (editable,                        │
  │                   filterable, exportable)                    │
  │                                                              │
  └── SQL_langrapgh.agent.invoke(history)                       │
           │                                                     │
           ▼  LangGraph state machine                           │
       classify_query                                           │
           │                                                     │
     ┌─────┼─────┐                                             │
     ▼     ▼     ▼                                             │
    sql  jira combined                                          │
     │     │     │                                             │
     ▼     │     ▼                                             │
  list_tables   generate_query ◄────────────────────────────── │
     ▼               │                                         │
  get_schema    check_query (SQL validation)                   │
     │               │                                         │
     └──────────► run_query / run_jira                        │
                      │                                         │
                      ▼                                         │
                 Tool results → LLM → final answer             │
                                                               │
                      MySQL Database                           │
                   ┌──────────────┐  ┌──────────────┐        │
                   │ realtimedata │  │ jira_tickets │        │
                   │ (test runs)  │  │ (JIRA cache) │        │
                   └──────────────┘  └──────────────┘        │
                                              ▲               │
                                     jira_sync.py            │
                                     (run separately)         │
                                              │               │
                                        JIRA REST API        │
└─────────────────────────────────────────────────────────────┘
```

**Chat query lifecycle:**
1. User types question; sidebar filters are prepended as `[Filters: ...]`.
2. Full conversation history passed to `agent.invoke()`.
3. Agent classifies intent → routes through state machine nodes.
4. SQL queries are generated, validated, and executed against MySQL.
5. JIRA lookups call live JIRA tools or the local cache table.
6. LLM formats results as markdown tables; `app.py` converts them to DataFrames.
7. Tool-call trace stored in session state and shown in "Reasoning" expander.

## Key Concepts / Gotchas

**Test ID prefix mismatch** — The `jira_tickets` table stores test IDs with a prefix like `CONF+TMO+<testid>`, while `realtimedata` stores bare IDs. Joins use `LIKE '%<testid>%'` rather than equality. The agent prompt strips the `TC_<digits> _` prefix before issuing joins.

**Non-standard datetime format** — Timestamps in `realtimedata` use `YYYY-MM-DD:HH:MM:SS` (colon between date and time, not a space). The agent prompt explicitly documents this so generated SQL uses the correct format.

**QGenie API key at runtime** — The key is entered by the user in the browser and set via `os.environ` for the session. It is never read from `.env`. On HTTP 401, the app clears the key and prompts again.

**JIRA scope** — All JIRA queries are scoped to `project = MSTCONF AND component in (4G_CAT, 5G_CAT)`. Queries outside this scope will return no results.

**No requirements.txt** — This is a backup snapshot. You must install dependencies manually (see Installation above) or recreate a `requirements.txt` with `pip freeze` after installing.

**Streamlit cache invalidation** — Dashboard data is cached for 300 s via `@st.cache_data(ttl=300)`. After an inline row edit, `st.cache_data.clear()` is called explicitly to show updated data immediately.
