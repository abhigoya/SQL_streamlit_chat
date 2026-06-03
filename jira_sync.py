import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone

import pymysql
import pymysql.cursors
from atlassian import Jira
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 200
PAGE_SIZE = 100

JIRA_FIELDS = [
    "key", "summary", "status", "reporter", "description",
    "customfield_37546",   # test_id
    "customfield_37543",   # cr
    "customfield_39515",   # pl
    "customfield_11743",   # crm_build_id
    "customfield_40467",   # modem_area
    "customfield_12830",   # resolution_notes
    "customfield_39568",   # operator
    "created", "updated",
]

DDL = """
CREATE TABLE IF NOT EXISTS jira_tickets (
    ticket_key       VARCHAR(50)   PRIMARY KEY,
    summary          TEXT          NOT NULL,
    status           VARCHAR(100),
    reporter         VARCHAR(255),
    description      TEXT,
    cr               VARCHAR(255),
    test_id          VARCHAR(255),
    crm_build_id     VARCHAR(255),
    modem_area       VARCHAR(255),
    pl               VARCHAR(255),
    resolution_notes TEXT,
    operator         VARCHAR(100),
    created_at       DATETIME,
    updated_at       DATETIME,
    synced_at        DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_test_id    (test_id(191)),
    INDEX idx_operator   (operator),
    INDEX idx_updated    (updated_at),
    INDEX idx_status     (status),
    INDEX idx_modem_area (modem_area)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def get_db_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        db=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def get_jira_client():
    return Jira(
        url=os.getenv("JIRA_URL"),
        username=os.getenv("JIRA_USERNAME"),
        password=os.getenv("JIRA_PASSWORD"),
        verify_ssl=False,
    )


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    log.info("jira_tickets table ready")


def _parse_dt(value):
    if not value:
        return None
    try:
        # ISO8601 with tz offset: 2026-04-24T04:42:37.000-0700
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def _upsert_sql(rows):
    cols = [
        "ticket_key", "summary", "status", "reporter", "description",
        "cr", "test_id", "crm_build_id", "modem_area", "pl",
        "resolution_notes", "operator", "created_at", "updated_at",
    ]
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c}=VALUES({c})" for c in cols if c != "ticket_key")
    sql = (
        f"INSERT INTO jira_tickets ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )
    values = [
        [row.get(c) for c in cols]
        for row in rows
    ]
    return sql, values


def upsert_batch(conn, rows):
    if not rows:
        return
    sql, values = _upsert_sql(rows)
    with conn.cursor() as cur:
        cur.executemany(sql, values)
    conn.commit()


# ── CSV load ─────────────────────────────────────────────────────────────────

def load_from_csv(conn, path):
    log.info("Loading from CSV: %s", path)
    batch = []
    total = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            created = _parse_dt(row.get("Created"))
            batch.append({
                "ticket_key":       row.get("JIRA ID", "").strip() or None,
                "summary":          (row.get("Summary") or "").strip(),
                "status":           (row.get("Status") or "").strip() or None,
                "reporter":         (row.get("Reporter") or "").strip() or None,
                "description":      (row.get("Description") or "").strip() or None,
                "cr":               (row.get("CRs") or "").strip() or None,
                "test_id":          (row.get("Test Case ID") or "").strip() or None,
                "crm_build_id":     (row.get("CRM Build Id") or "").strip() or None,
                "modem_area":       (row.get("Modem Area") or "").strip() or None,
                "pl":               (row.get("PL") or "").strip() or None,
                "resolution_notes": (row.get("Resolution Notes") or "").strip() or None,
                "operator":         (row.get("Operator") or "").strip() or None,
                "created_at":       created,
                "updated_at":       created,   # no Updated col in CSV; use created_at so incremental sync works
            })
            if not batch[-1]["ticket_key"]:
                batch.pop()
                continue
            if len(batch) >= BATCH_SIZE:
                upsert_batch(conn, batch)
                total += len(batch)
                log.info("  inserted %d rows so far…", total)
                batch = []
    if batch:
        upsert_batch(conn, batch)
        total += len(batch)
    log.info("CSV load complete — %d rows upserted", total)


# ── JIRA API sync ─────────────────────────────────────────────────────────────

def get_last_sync_time(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(updated_at) AS last FROM jira_tickets")
        row = cur.fetchone()
    val = row["last"] if row else None
    return val  # datetime or None


def _str_field(value):
    """Coerce a JIRA field value (string, select-list dict, multi-select list) to str or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        # Single select: {"value": "TMO", "id": "..."} or {"name": "...", "id": "..."}
        return value.get("value") or value.get("name") or None
    if isinstance(value, list):
        # Multi-select: [{"value": "a"}, {"value": "b"}]
        parts = [_str_field(v) for v in value]
        return ", ".join(p for p in parts if p) or None
    return str(value) or None


def _parse_issue(issue):
    f = issue.get("fields", {})
    return {
        "ticket_key":       issue.get("key"),
        "summary":          (f.get("summary") or ""),
        "status":           (f.get("status") or {}).get("name"),
        "reporter":         ((f.get("reporter") or {}).get("displayName")),
        "description":      _str_field(f.get("description")),
        "cr":               _str_field(f.get("customfield_37543")),
        "test_id":          _str_field(f.get("customfield_37546")),
        "crm_build_id":     _str_field(f.get("customfield_11743")),
        "modem_area":       _str_field(f.get("customfield_40467")),
        "pl":               _str_field(f.get("customfield_39515")),
        "resolution_notes": _str_field(f.get("customfield_12830")),
        "operator":         _str_field(f.get("customfield_39568")),
        "created_at":       _parse_dt(f.get("created")),
        "updated_at":       _parse_dt(f.get("updated")),
    }


def _upsert_api_sql(rows):
    """Upsert rows fetched from the JIRA API, including modem_area, operator, and resolution_notes."""
    cols = [
        "ticket_key", "summary", "status", "reporter", "description",
        "cr", "test_id", "crm_build_id", "modem_area", "pl",
        "resolution_notes", "operator", "created_at", "updated_at",
    ]
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(
        f"{c}=VALUES({c})"
        for c in cols if c != "ticket_key"
    )
    sql = (
        f"INSERT INTO jira_tickets ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )
    values = [[row.get(c) for c in cols] for row in rows]
    return sql, values


def _fetch_and_upsert(conn, jira, jql):
    start = 0
    total_fetched = 0
    while True:
        result = jira.jql_get_list_of_tickets(
            jql,
            fields=JIRA_FIELDS,
            start=start,
            limit=PAGE_SIZE,
        )
        issues = result if isinstance(result, list) else result.get("issues", [])
        if not issues:
            break
        batch = [_parse_issue(i) for i in issues]
        sql, values = _upsert_api_sql(batch)
        with conn.cursor() as cur:
            cur.executemany(sql, values)
        conn.commit()
        total_fetched += len(issues)
        log.info("  fetched & upserted %d tickets (total so far: %d)", len(issues), total_fetched)
        start += len(issues)
        # Some JIRA DC responses embed total; stop if we know we're done
        if isinstance(result, dict):
            server_total = result.get("total", 0)
            if start >= server_total:
                break
        if len(issues) < PAGE_SIZE:
            break
    return total_fetched


def run_full_sync(conn, jira):
    log.info("Starting FULL sync (5 years)…")
    jql = (
        "project = MSTCONF AND component in (4G_CAT, 5G_CAT) "
        "AND created >= -1825d ORDER BY updated ASC"
    )
    total = _fetch_and_upsert(conn, jira, jql)
    log.info("Full sync complete — %d tickets processed", total)


def run_incremental(conn, jira):
    last = get_last_sync_time(conn)
    if last:
        since = last.strftime("%Y-%m-%d %H:%M")
        log.info("Incremental sync since %s…", since)
        jql = (
            f"project = MSTCONF AND component in (4G_CAT, 5G_CAT) "
            f"AND updated >= '{since}' ORDER BY updated ASC"
        )
    else:
        log.info("No updated_at found — falling back to last 2 days…")
        jql = (
            "project = MSTCONF AND component in (4G_CAT, 5G_CAT) "
            "AND updated >= -7d ORDER BY updated ASC"
        )
    total = _fetch_and_upsert(conn, jira, jql)
    log.info("Incremental sync complete — %d tickets processed", total)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sync JIRA tickets to MySQL jira_tickets table")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--load-csv", metavar="PATH", help="Load initial data from CSV file")
    group.add_argument("--full-sync", action="store_true", help="Full 5-year sync via JIRA API")
    args = parser.parse_args()

    conn = get_db_conn()
    create_table(conn)

    if args.load_csv:
        load_from_csv(conn, args.load_csv)
    elif args.full_sync:
        jira = get_jira_client()
        run_full_sync(conn, jira)
    else:
        jira = get_jira_client()
        run_incremental(conn, jira)

    conn.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
