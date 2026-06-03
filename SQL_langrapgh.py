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
    include_tables=["realtimedata", "jira_tickets"],
)

# print(f"Dialect: {db.dialect}")
# print(f"Available tables: {db.get_usable_table_names()}")
# print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')

llm = QGenieChat(model = "anthropic::claude-4-5-sonnet")


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
For setup health / UE issue queries:

The key pattern to detect: TE logs are present (TEBuild populated, testresult has value)
but UE log is absent — indicated by ue_build = '0' (string, NOT numeric 0).
ue_build is CHAR(255) NOT NULL DEFAULT '0'; never use IS NULL to test it.
UE_log_path is also CHAR(255) NOT NULL — never use IS NULL on it either.
Use ue_build = '0' for "no UE log" and ue_build != '0' for "has UE log".

--- AUTOMATIC TIME SCOPING (applies to Cases A and B only — NOT Case C) ---

When the user does NOT specify an explicit time window, automatically apply the relevant
monitoring window based on the current day and time. Use MySQL's DAYOFWEEK(NOW()) and
HOUR(NOW()) to determine the window. DAYOFWEEK: 1=Sunday, 2=Mon, 3=Tue, 4=Wed, 5=Thu,
6=Fri, 7=Saturday.

WEEKDAY (DAYOFWEEK IN (2,3,4,5,6) — Monday through Friday):
  Setup issues are expected during the unmonitored overnight period (5 PM to 10 AM).
  Determine <window_start> and <window_end> as follows:
  - HOUR(NOW()) < 17 → overnight window has ended for today:
      window_start = CONCAT(DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 DAY), '%Y-%m-%d'), ':17:00:00')
      window_end   = CONCAT(DATE_FORMAT(CURDATE(), '%Y-%m-%d'), ':10:00:00')
      Apply: AND starttime >= window_start AND starttime <= window_end
  - HOUR(NOW()) >= 17 → overnight window just started:
      window_start = CONCAT(DATE_FORMAT(CURDATE(), '%Y-%m-%d'), ':17:00:00')
      Apply: AND starttime >= window_start   (no end constraint — night is in progress)

WEEKEND (DAYOFWEEK IN (7,1) — Saturday or Sunday):
  Automation can run any time; no one is monitoring. Cover the full weekend from Friday 17:00.
  - DAYOFWEEK(NOW()) = 7 (Saturday):
      window_start = CONCAT(DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 DAY), '%Y-%m-%d'), ':17:00:00')
  - DAYOFWEEK(NOW()) = 1 (Sunday):
      window_start = CONCAT(DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 2 DAY), '%Y-%m-%d'), ':17:00:00')
  Apply: AND starttime >= window_start   (no end constraint)

Always state the window used in the "**Diagnosis:**" summary line, e.g.:
  "Checking overnight window: 2026-05-26:17:00:00 → 2026-05-27:10:00:00" (weekday)
  "Checking full weekend window from: 2026-05-23:17:00:00" (weekend)

--- CASE A: User asks about ALL setups (no specific setup named, no time filter) ---

Run this SINGLE query — replace the static NOW()-INTERVAL filter with the automatic window above:

  SELECT rd.setupname,
         COUNT(*) AS tests_without_ue_log,
         MIN(rd.starttime) AS issue_started_at,
         MAX(rd.starttime) AS latest_test_time
  FROM realtimedata rd
  LEFT JOIN (
    SELECT setupname, MAX(starttime) AS last_ue_log_time
    FROM realtimedata
    WHERE ue_build != '0'
    GROUP BY setupname
  ) lu ON rd.setupname = lu.setupname
  WHERE rd.starttime > COALESCE(lu.last_ue_log_time, '2000-01-01')
    AND rd.starttime >= <window_start>
    [AND rd.starttime <= <window_end>  -- only on weekday before-5PM case]
  GROUP BY rd.setupname
  HAVING COUNT(*) > 0
  ORDER BY tests_without_ue_log DESC

Present results as a table: setupname | tests_without_ue_log | issue_started_at | latest_test_time
If 0 rows → "No UE issues detected in the [overnight / weekend] window."

--- CASE B: User asks about a SPECIFIC setup with NO time window filter ---
(Use this only when user is asking about the CURRENT state: "is there an issue now?")

CRITICAL: The inner subquery finding the last good UE log MUST remain unfiltered by time
(it needs all history to correctly anchor the detection point).
Apply the automatic time window ONLY to the outer WHERE clauses in Query 2 and Query 3.

Query 1 — Find when UE logs last saved (NO starttime filter — unchanged):
  SELECT starttime AS last_ue_log_time, testid AS last_test_with_ue_log
  FROM realtimedata
  WHERE setupname LIKE '%<setup>%'
    AND ue_build != '0'
  ORDER BY starttime DESC
  LIMIT 1

Query 2 — Count tests without UE log in the monitoring window:
  SELECT COUNT(*) AS tests_without_ue_log,
         MIN(starttime) AS issue_started_at,
         MAX(starttime) AS latest_test_time
  FROM realtimedata
  WHERE setupname LIKE '%<setup>%'
    AND starttime > (
      SELECT COALESCE(MAX(starttime), '2000-01-01')
      FROM realtimedata
      WHERE setupname LIKE '%<setup>%'
        AND ue_build != '0'
      /* NO starttime filter here — must look at all history */
    )
    AND starttime >= <window_start>
    [AND starttime <= <window_end>  -- only on weekday before-5PM case]

Query 3 — Recent test detail within the monitoring window:
  SELECT testid, testresult, starttime, TEBuild, ue_build, UE_log_path
  FROM realtimedata
  WHERE setupname LIKE '%<setup>%'
    AND starttime >= <window_start>
    [AND starttime <= <window_end>  -- only on weekday before-5PM case]
  ORDER BY starttime DESC
  LIMIT 10

Diagnosis rules (Case B):
- tests_without_ue_log >= 1 → UE issue is ongoing in the monitoring window
  → "UE issue detected since <issue_started_at>. <N> test cases ran without UE logs."
- tests_without_ue_log = 0 → No issue in the monitoring window
  → "No setup issue detected in the [overnight / weekend] window. Last UE log saved at <last_ue_log_time>."

--- CASE C: User has an explicit time window filter OR is asking about a past period ---
(Use this when user specifies starttime >= X, or asks "were there issues", "what happened",
 or points to a specific past timestamp — the issue may have already recovered.)

Even though the user specifies a time window, ALWAYS apply the monitored-hours filter
within that window — only rows during unmonitored periods are relevant:
  - Weekend rows (DAYOFWEEK 1=Sunday or 7=Saturday): any hour
  - Weekday rows (DAYOFWEEK 2–6): overnight only — hour >= 17 OR hour < 10

NOTE: starttime is stored as 'YYYY-MM-DD:HH:MM:SS' (colon between date and time).
Extract hour as: CAST(SUBSTRING(starttime, 12, 2) AS UNSIGNED)
Extract date as: SUBSTRING(starttime, 1, 10)  -- valid input for DAYOFWEEK()

Run this gaps-and-islands query to find ALL streaks of missing UE logs in the window:

  SELECT MIN(starttime) AS streak_started,
         MAX(starttime) AS streak_ended,
         COUNT(*) AS tests_no_ue_log
  FROM (
    SELECT starttime,
           SUM(CASE WHEN ue_build != '0' THEN 1 ELSE 0 END)
             OVER (ORDER BY starttime ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS good_ue_count
    FROM realtimedata
    WHERE setupname LIKE '%<setup>%'
      AND starttime >= '<user_starttime_filter>'
      AND (
        -- Weekend: any hour counts
        DAYOFWEEK(SUBSTRING(starttime, 1, 10)) IN (1, 7)
        OR
        -- Weekday: overnight hours only (17:00–09:59)
        (
          DAYOFWEEK(SUBSTRING(starttime, 1, 10)) BETWEEN 2 AND 6
          AND (
            CAST(SUBSTRING(starttime, 12, 2) AS UNSIGNED) >= 17
            OR CAST(SUBSTRING(starttime, 12, 2) AS UNSIGNED) < 10
          )
        )
      )
  ) t
  WHERE ue_build = '0'
  GROUP BY good_ue_count
  HAVING COUNT(*) >= 5
  ORDER BY streak_started

This finds every contiguous streak of 5+ consecutive monitored-hours tests without UE logs.
Present as: streak_started | streak_ended | tests_no_ue_log
If 0 rows → "No UE issues (streaks of 5+ consecutive missing UE logs) found in this period
             during monitored hours (weekday nights 17:00–10:00 and full weekends)."

Always present:
1. A bold "**Diagnosis:** ..." summary line first (include the monitoring window used)
2. Then the streak table or detail table
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

