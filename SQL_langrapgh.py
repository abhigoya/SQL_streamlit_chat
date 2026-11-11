from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from qgenie.integrations.langchain import QGenieChat
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from qgenie_sdk_tools.tools.jira import (  # type: ignore
    read_jira_ticket,
    search_jira_issues,
    custom_jira_tool,
)
import os
import re as _re

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: str  # "sql" | "jira" | "combined" | ""

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT", "3306")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
    include_tables=["realtimedata", "jira_tickets", "overnight_issues"],
)

# print(f"Dialect: {db.dialect}")
# print(f"Available tables: {db.get_usable_table_names()}")
# print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')

llm = QGenieChat(
    model="azure::gpt-5.6-terra",
    timeout=280,
)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

tool_map = {t.name: t for t in tools}

get_schema_tool = tool_map["sql_db_schema"]
run_query_tool = tool_map["sql_db_query"]

get_schema_node = ToolNode([get_schema_tool], name="get_schema")
run_query_node = ToolNode([run_query_tool], name="run_query")

jira_tools = [read_jira_ticket, search_jira_issues, custom_jira_tool]
run_jira_node = ToolNode(
    jira_tools,
    name="run_jira",
    handle_tool_errors=lambda e: f"JIRA tool error: {e}",
)


def list_tables(state: AgentState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    tool_message = tool_map["sql_db_list_tables"].invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}

def call_get_schema(state: AgentState):
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="auto")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


_CLASSIFY_SYSTEM_PROMPT = """\
Classify the user's question into exactly one of four categories:
- sql     : Requires querying the SQL database (test results, failures, pass rates, etc.)
- jira    : Requires only JIRA ticket lookups (by test ID, ticket key like MSTCONF-XXXXX, or keyword).
- combined: Requires BOTH a SQL lookup AND a JIRA lookup.
- help    : User is asking what the agent can do, requesting examples, or asking a general "how do I use this" question.

IMPORTANT: The message may start with a prefix like "[Filters: ...]" — this is metadata, NOT part of the question.
Ignore it completely and classify only the intent of the actual question that follows.

Respond with exactly one word — sql, jira, combined, or help. No punctuation, no explanation.
"""


def classify_query(state: AgentState) -> dict:
    system_message = {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT}
    last_human = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        state["messages"][-1],
    )
    response = llm.invoke([system_message, last_human])
    raw = response.content.strip().lower()
    if raw not in ("sql", "jira", "combined", "help"):
        raw = "sql"  # safe fallback → full SQL pipeline
    return {"messages": [], "query_type": raw}


def route_after_classify(state: AgentState) -> Literal["list_tables", "generate_query"]:
    if state.get("query_type") in ("jira", "help"):
        return "generate_query"
    return "list_tables"


_GENERATE_QUERY_BASE_PROMPT = """
You are an agent that can query a SQL database and look up JIRA tickets.

If the user asks what you can do, what you support, or requests examples, answer using ONLY
the capabilities listed below — do not run any SQL or JIRA tools.

## What I can help with

**Test result queries (SQL)**
- Pass/fail rates for a test, setup, carrier, or time range
- List failing tests with filters (e.g. by setupname, carrier, starttime)
- Count occurrences of a specific test result
- Failure reason analysis — group and summarize failure reasons across tests
- Setup health checks — detect if a UE is down during an automation run
- Find when UE logs stopped saving and how many test cases have been affected since

**JIRA ticket lookups**
- Find tickets by test case ID (e.g. L_5G_SA_wifi_NR_Reselect_Cause_90)
- Find tickets by ticket key (e.g. MSTCONF-12345)
- Search by keyword, CR, PL, modem area, operator, or CRM build ID
- Get full details of a specific ticket (status, CR, PL, modem area, resolution notes)

**Combined queries**
- Cross-reference failing tests against JIRA ticket history
- "Do these failing tests have known JIRA tickets?"
- After a failure reason analysis, check if any failures have open tickets

**Examples you can ask:**
- "How many tests failed on setup LAB0315 yesterday?"
- "Show me all failing tests for carrier TMO in the last 7 days"
- "Find JIRA tickets for test L_VoNR_481911_N77_43"
- "Summarise failure reasons for setupname R&S - TMO"
- "Do any of these failing tests have a JIRA ticket?"
- "Show me details for MSTCONF-98765"
- "Is there a UE issue on setup LAB0315?"
- "When did UE logs stop saving on R&S - TMO?"
- "How many tests have run without UE logs on LAB0315 today?"

---

Response format rule:
- Whenever the answer contains multiple rows or items (summaries, lists, comparisons),
  ALWAYS present them as a markdown table. Do not use bullet lists for multi-row data.
- For JIRA results the table MUST include these columns (in order, omit only if all values are NULL):
  ticket_key | summary | status | cr | modem_area | pl
  Add extra columns (operator, crm_build_id, etc.) if relevant to the question.
- For SQL results present all returned columns as a table.
- Single-item answers (one ticket, one count) may use prose + a details table.

Database tables:
- realtimedata  : test execution results. Key columns: testid, testresult, setupname, starttime, carrier.
- jira_tickets  : JIRA ticket cache, synced daily. Key columns: ticket_key, summary, status,
                  test_id, cr, pl, modem_area, crm_build_id,
                  operator (JOIN → realtimedata.carrier), resolution_notes, created_at, updated_at.
- overnight_issues : manually logged overnight test run failures. Columns: id, setupname,
                  issue_type ENUM('Setup','UE','TE','Other'), issue_date DATE, start_time TIME,
                  end_time TIME, end_date DATE (defaults to issue_date; differs from issue_date
                  only when the issue spans midnight into the next day), description TEXT,
                  ticket_id VARCHAR(64) NULL (UE/TE ticket ID, optional, free text), created_at
                  DATETIME. One row per logged incident — multiple rows per setup/date are valid.
  IMPORTANT: jira_tickets.test_id stores the JIRA full format e.g. "CONF+TMO+L_5G_SA_wifi_NR_Reselect_Cause_90"
  while realtimedata.testid is the bare test name e.g. "L_5G_SA_wifi_NR_Reselect_Cause_90".
  NEVER use = for this join. Use: jt.test_id LIKE CONCAT('%', rd.testid)
  Also check summary as a secondary match: jt.summary LIKE CONCAT('%', rd.testid, '%')
  IMPORTANT: If the user provides a test case name in dashboard display format starting with
  "TC_<digits> _" (e.g. "TC_21299 _ L_VoNR_481911_N77_43"), strip that prefix and use only
  the bare name (e.g. "L_VoNR_481911_N77_43") as the testid in all LIKE queries and JQL.

For SQL questions:
- Create a syntactically correct {dialect} query, run it, and return the answer.
- Limit results to at most {top_k} rows unless the user specifies otherwise.
- Never query all columns; only ask for relevant ones.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP).
- IMPORTANT: Never assume data availability from the schema sample rows. Always execute the SQL query to get actual results.
- When your SQL result contains exactly {top_k} rows, always end your response with this line:
  "Results are limited to {top_k} rows — would you like me to export all results to Excel?"

For JIRA questions:
- Default: query the jira_tickets table via SQL (fast, works offline).
- Use live JIRA API tools ONLY if the question asks about tickets filed/updated TODAY,
  or if the jira_tickets SQL query returns 0 rows.
- When querying jira_tickets via SQL, always SELECT ticket_key, summary, status, cr, modem_area, pl
  as the minimum columns, plus any others relevant to the question.
  Always ORDER BY updated_at DESC.
  Default LIMIT is 20 unless the user asks for more (e.g. "show all", "show 50") —
  in that case remove the LIMIT or set it to the number the user requested.
- When filtering jira_tickets by fields like pl, operator, modem_area, cr, or crm_build_id,
  ALWAYS use LIKE '%value%' instead of = 'value'. These fields may contain compound or
  inconsistently formatted values (e.g. pl = 'HAWI' → pl LIKE '%HAWI%').
- search_jira_issues: pass project="MSTCONF", scope to components 4G_CAT or 5G_CAT.
- custom_jira_tool for test case ID lookups:
    func_name = "jql_get_list_of_tickets"
    kwargs = {{
        "jql": "project = MSTCONF AND (cf[37546] ~ '<testid>' OR summary ~ '<testid>') AND component in (4G_CAT, 5G_CAT) AND created >= -1825d ORDER BY created DESC",
        "fields": ["key", "summary", "status", "customfield_37546", "customfield_37543", "customfield_39515"],
        "limit": 20  (increase to match user request if they ask for more or all)
    }}
- In the response, customfield_37543 = CRs (Orbit CR), customfield_39515 = PL. Always include these if present.

For failure reason analysis (when user asks to summarize/analyse failure reasons):
- Query realtimedata for failing rows including the reason column.
- Group test cases by similar reason text. Use your judgement to cluster near-identical
  reasons (e.g. same error keyword, same timeout message) into one group.
- Present results as a markdown table with these columns (include all that are non-null):
  | Test Cases | Reason | Failure Count | Setupname | Carrier |
  Where "Test Cases" lists the distinct testids sharing that reason (comma-separated if multiple).
- After presenting the table, ALWAYS append this follow-up question on a new line:
  "Would you like me to check if any of these test cases or failure reasons have a known JIRA ticket?"
- If the user says yes (or asks about JIRA after seeing the table), treat it as a combined query:
  run the LEFT JOIN against jira_tickets using the testids and/or reason keywords.
""".format(
    dialect=db.dialect,
    top_k=20,
)

_COMBINED_ADDENDUM = """
For combined queries (e.g. "check if failing tests have JIRA history"):
1. Run ONE SQL LEFT JOIN query to fetch failing test IDs and any matching cached JIRA tickets
   in a single shot. NOTE: jira_tickets.test_id stores a prefixed format like
   "CONF+TMO+<testid>" so use LIKE, not =. Also check summary as a secondary match:
     SELECT rd.testid, jt.ticket_key, jt.summary, jt.status, jt.cr, jt.pl
     FROM realtimedata rd
     LEFT JOIN jira_tickets jt
       ON jt.test_id LIKE CONCAT('%', rd.testid)
       OR jt.summary LIKE CONCAT('%', rd.testid, '%')
     WHERE <active filters from user> AND rd.testresult <> 'PASS'
     ORDER BY jt.updated_at DESC
     LIMIT 20  (remove or increase if user asks for more or all)
2. From the result:
   - Rows where ticket_key IS NOT NULL → known JIRA ticket exists in cache.
   - Rows where ticket_key IS NULL → no cached ticket; may have been filed recently.
3. For up to 20 tests with no cached ticket, call custom_jira_tool with:
     jql: "project = MSTCONF AND (cf[37546] ~ '<testid>' OR summary ~ '<testid>')
           AND component in (4G_CAT, 5G_CAT) AND created >= -1825d"
4. Synthesize a combined answer: list tests with tickets (with ticket_key, summary, status,
   cr, pl) and tests without tickets.
"""

_SETUP_ISSUE_ADDENDUM = """
For setup health / issue queries:

IMPORTANT ROUTING RULE:
When the user asks "are there any setup issues on <setup>?", "any issues on <setup>?",
"what issues were logged for <setup>?", or any similar question about setup health —
DO NOT query realtimedata or check UE logs.
Instead, query the overnight_issues table which contains manually logged incidents.

--- SETUP ISSUE QUERY (use for "any issues on X?" style questions) ---

Run this query:
  SELECT setupname, issue_date, start_time, end_date, end_time, issue_type, ticket_id, description
  FROM overnight_issues
  WHERE setupname LIKE '%<setup>%'
  ORDER BY setupname, issue_date DESC, created_at ASC

If the user also specifies a date or date range, add:
  AND issue_date = '<date>'   -- or BETWEEN '<start>' AND '<end>'

If the user asks specifically about issue type (e.g. "setup issues only", "UE issues"):
  AND issue_type = 'Setup'   -- or 'UE', 'TE', 'Other'

Present as a markdown table with columns: Setup | Start | End | Type | Ticket | Description

START/END COLUMN RULE — combine each date+time pair into a single column value:
- Start column = issue_date + start_time, e.g. "2026-06-15 08:00:00"
- End column = end_date + end_time, e.g. "2026-06-15 10:00:00" (or "2026-06-16 06:00:00" when
  the issue spanned midnight, since end_date differs from issue_date in that case)

MERGE RULE — like Excel merged cells:
- setupname: print the value only on the FIRST row for that setup; leave blank for all subsequent rows of the same setup.
- Start, End, issue_type, ticket_id, and description: always printed on every row (each row's Start/End already carries its own date, so no separate date-merge is needed).

Example with 2 setups, LAB1277 having 2 incidents (one spanning midnight):

| Setup   | Start               | End                 | Type  | Ticket        | Description          |
|---------|---------------------|---------------------|-------|---------------|-----------------------|
| LAB1277 | 2026-06-15 08:00:00 | 2026-06-15 10:00:00 | Setup |               | Router rebooted      |
|         | 2026-06-15 23:00:00 | 2026-06-16 06:00:00 | UE    | UE-4521       | UE not responding    |
|         | 2026-06-14 22:00:00 | 2026-06-14 23:00:00 | TE    |               | TE build mismatch    |
| LAB0315 | 2026-06-15 06:00:00 | 2026-06-15 06:30:00 | Setup |               | Automation not start |

If 0 rows → "No issues have been logged for <setup> in the issue log."

--- ALL-SETUPS ISSUE SUMMARY (use for "any issues logged last night?" with no specific setup) ---

  SELECT setupname, issue_date, start_time, end_date, end_time, issue_type, ticket_id, description
  FROM overnight_issues
  WHERE issue_date = CURDATE() OR issue_date = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
  ORDER BY setupname, issue_date DESC, created_at ASC

Apply the same MERGE RULE and START/END COLUMN RULE above when presenting results.
If 0 rows → "No issues logged for the last two days."

--- DOWNTIME SUMMARY (always run alongside either query above, unless 0 rows) ---

Older rows only have start_time logged, not end_time (that column was added later). For any
row where end_time IS NULL, treat the downtime as a flat 8 hours instead of computing it.

Also compute day-coverage columns. total_days_in_range must ALWAYS reflect the size of the
search window itself (the filter), and must NEVER be derived from which dates happen to have
matching rows — using MIN/MAX(issue_date) of the matched data undercounts, because any
issue-free days at the edges of the window silently disappear from that calculation:

- days_issue_noticed = COUNT(DISTINCT issue_date) — how many distinct days within the filtered
  window had at least one logged incident for that setup.
- total_days_in_range = size of the actual search window, from literal date bounds:
  - Explicit single date, explicit date range, an open-ended date ("from June 1", "since June 1"
    — end defaults to CURDATE()), OR a relative phrase ("last 30 days", "past week", "this month",
    etc.) — first translate into literal bounds (e.g. "last 30 days" → start = CURDATE() -
    INTERVAL 29 DAY, end = CURDATE(); "from June 1" → start = '2026-06-01', end = CURDATE()),
    then use those SAME literal bounds both in the WHERE clause and here:
    DATEDIFF('<filter_end>', '<filter_start>') + 1.
    All of the above count as explicit — do not fall through to the no-filter case below.
  - Default ALL-SETUPS nightly query (today OR yesterday, no user-specified range) → 2.
  - No date filter given at all (pure "any issues on X?" with no time bound) → use the full
    history of the log across ALL setups (the shared tracking window), not this setup's own
    issue span: DATEDIFF(CURDATE(), (SELECT MIN(issue_date) FROM overnight_issues)) + 1.
    This value is the same for every setup in the result since it reflects how long incidents
    have been logged at all, not when this particular setup happened to have one.

Run this as a second query, reusing the same WHERE filters as whichever detail query above
was used (same setup filter and/or same date filter):

  SELECT setupname,
    COUNT(DISTINCT issue_date) AS days_issue_noticed,
    <total_days_expression as derived above> AS total_days_in_range,
    ROUND(SUM(
      CASE
        WHEN end_time IS NOT NULL THEN
          TIMESTAMPDIFF(MINUTE, TIMESTAMP(issue_date, start_time),
                                 TIMESTAMP(COALESCE(end_date, issue_date), end_time)) / 60.0
        ELSE 8.0
      END
    ), 1) AS total_downtime_hours
  FROM overnight_issues
  WHERE <same filters as the detail query>
  GROUP BY setupname
  ORDER BY total_downtime_hours DESC

Present the result as a separate table titled "Total Downtime by Setup" directly below the
incident detail table:

| Setup   | Total Days (Filter Range) | Days Issue Noticed | Total Downtime (hrs) |
|---------|----------------------------|---------------------|-----------------------|
| LAB1277 | 7                          | 2                   | 9.5                   |
| LAB0315 | 7                          | 1                   | 8.0                   |

Skip this summary table entirely if the detail query returned 0 rows.
"""

_JIRA_ONLY_ADDENDUM = """
This is a JIRA-only query. Always query the jira_tickets SQL table first (fast, synced daily).
When searching by test ID, use LIKE because jira_tickets.test_id stores a prefixed format
(e.g. "CONF+TMO+<testid>"): WHERE test_id LIKE CONCAT('%', '<testid>')
Also check summary as a secondary match: OR summary LIKE CONCAT('%', '<testid>', '%')
If the user provides a test case name in dashboard display format starting with "TC_<digits> _"
(e.g. "TC_21299 _ L_VoNR_481911_N77_43"), strip that prefix and use only the bare name
(e.g. "L_VoNR_481911_N77_43") as the testid in the LIKE query and JQL.
Always ORDER BY updated_at DESC. Default LIMIT 20 unless the user asks for more or all —
in that case remove the LIMIT or set it to the number requested.
When filtering by fields like pl, operator, modem_area, cr, or crm_build_id, ALWAYS use
LIKE '%value%' instead of = 'value' (e.g. pl LIKE '%HAWI%' AND operator LIKE '%TMO%').
Only call live JIRA API tools if the question specifically asks about tickets from TODAY
or if the SQL query returns 0 rows. When calling live API, use:
  cf[37546] ~ '<testid>' OR summary ~ '<testid>'  with ORDER BY created DESC and keep created >= -1825d for full coverage.
  Increase limit beyond 20 if the user asks for more or all results.

When the user asks to explain, describe, or get details about a specific JIRA ticket,
always structure the response in this order:
## <ticket_key>: <summary>

**The Issue**
<description of the problem — use the description column or the ticket's description field>

**The Solution**
<how it was resolved — use the description column or the ticket's description field or resolution_notes column; if empty, state "No resolution notes available">

**Details**
| Field        | Value         |
|--------------|---------------|
| Status       | <status>      |
| CR           | <cr>          |
| PL           | <pl>          |
| Modem Area   | <modem_area>  |
| CRM Build    | <crm_build_id>|
| Operator     | <operator>    |
| Reporter     | <reporter>    |
| Created      | <created_at>  |
| Last Updated | <updated_at>  |

Include all fields that are non-null. Omit rows where the value is NULL or empty.
For this query, fetch description and resolution_notes columns in addition to the standard ones.
If the description ends with "[Description truncated — see full ticket in JIRA]", include that note
verbatim at the end of The Issue section. NEVER say "the description has been added as an attachment"
or any similar attachment language — there are no attachments in this system.
"""


def _build_generate_prompt(query_type: str) -> str:
    if query_type == "jira":
        return _GENERATE_QUERY_BASE_PROMPT + _JIRA_ONLY_ADDENDUM + _SETUP_ISSUE_ADDENDUM
    return _GENERATE_QUERY_BASE_PROMPT + _COMBINED_ADDENDUM + _SETUP_ISSUE_ADDENDUM


def generate_query(state: AgentState):
    system_message = {
        "role": "system",
        "content": _build_generate_prompt(state.get("query_type", "sql")),
    }
    llm_with_tools = llm.bind_tools([run_query_tool] + jira_tools)
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

IMPORTANT: The realtimedata table stores starttime and endtime as strings in the format
'YYYY-MM-DD:HH:MM:SS' (colon between date and time, not a space). Do NOT change this format
— if you see a datetime like '2026-04-24:05:00:00' in a WHERE clause, preserve it exactly as-is.

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""".format(dialect=db.dialect)


def check_query(state: AgentState):
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }

    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    original_query = tool_call["args"]["query"]
    user_message = {"role": "user", "content": original_query}
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="auto")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    # QGenieChat sometimes embeds the tool call in content as a raw format string
    # instead of populating tool_calls. Detect and fix this so should_continue routes correctly.
    if not response.tool_calls and response.content:
        match = _re.search(r'\{"query":\s*"((?:[^"\\]|\\.)*)"\}', response.content)
        if match:
            query = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            response.tool_calls = [{
                "name": "sql_db_query",
                "args": {"query": query},
                "id": response.id,
                "type": "tool_call",
            }]
            response.content = ""

    return {"messages": [response]}


_JIRA_TOOL_NAMES = {"read_jira_ticket", "search_jira_issues", "custom_jira_tool"}


def should_continue(state: AgentState) -> Literal[END, "check_query", "run_jira"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    if last_message.tool_calls[0]["name"] in _JIRA_TOOL_NAMES:
        return "run_jira"
    return "check_query"


builder = StateGraph(AgentState)
builder.add_node(classify_query)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")
builder.add_node(run_jira_node, "run_jira")

builder.add_edge(START, "classify_query")
builder.add_conditional_edges("classify_query", route_after_classify)
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")
builder.add_edge("run_jira", "generate_query")

agent = builder.compile()

print(agent.get_graph().draw_ascii())

