import json
import re
from sqlalchemy import text
from sqlalchemy import inspect

SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "sqlite_master", "mysql", "performance_schema", "sys"}


import json
import re
from sqlalchemy import text, inspect

SYSTEM_SCHEMAS = {
    "pg_catalog", "information_schema", "sqlite_master",
    "mysql", "performance_schema", "sys", "pg_toast"
}

def _normalize_schema(schema: str | None, dialect: str) -> str:
    """Return '' for default schema; hide SQLite's main/temp."""
    if not schema:
        return ""
    s = schema.strip()
    if dialect == "sqlite" and s.lower() in {"main", "temp"}:
        return ""
    return s

def _fq_name(schema: str, table: str) -> str:
    """Quote each identifier part separately."""
    return f'"{schema}"."{table}"' if schema else f'"{table}"'

def get_db_schema(engine, sample_rows: int = 3, max_tables: int = 50):
    insp = inspect(engine)
    dialect = engine.dialect.name  # 'sqlite', 'postgresql', 'mysql', ...
    # --- discover schemas ---
    try:
        if dialect == "sqlite":
            raw_schemas = [None]  # hide main/temp
        else:
            raw_schemas = [
                s for s in insp.get_schema_names()
                if s and s not in SYSTEM_SCHEMAS and not s.startswith("_")
            ] or [None]
    except Exception as e:
        print(f"[get_db_schema] get_schema_names failed: {e}")
        raw_schemas = [None]

    tables_meta = []
    count = 0

    with engine.connect() as conn:
        for raw_schema in raw_schemas:
            schema = _normalize_schema(raw_schema, dialect)

            # tables + views
            try:
                tbls = insp.get_table_names(schema=raw_schema) or []
            except Exception as e:
                print(f"[get_db_schema] get_table_names({raw_schema}) failed: {e}")
                tbls = []
            try:
                views = insp.get_view_names(schema=raw_schema) or []
            except Exception as e:
                print(f"[get_db_schema] get_view_names({raw_schema}) failed: {e}")
                views = []

            for t in (tbls + views):
                if count >= max_tables:
                    break

                # columns
                cols_meta = []
                try:
                    for c in insp.get_columns(t, schema=raw_schema) or []:
                        cols_meta.append({
                            "name": c.get("name"),
                            "type": str(c.get("type")),
                            "nullable": c.get("nullable", True),
                        })
                except Exception as e:
                    print(f"[get_db_schema] get_columns({t},{raw_schema}) failed: {e}")

                # FKs
                fks = []
                try:
                    for fk in insp.get_foreign_keys(t, schema=raw_schema) or []:
                        fks.append({
                            "constrained_columns": fk.get("constrained_columns", []),
                            "referred_table": fk.get("referred_table"),
                            "referred_schema": _normalize_schema(
                                fk.get("referred_schema"), dialect
                            ),
                        })
                except Exception as e:
                    print(f"[get_db_schema] get_foreign_keys({t},{raw_schema}) failed: {e}")

                # samples
                samples = []
                try:
                    fq = _fq_name(schema, t)
                    res = conn.execute(text(f"SELECT * FROM {fq} LIMIT {int(sample_rows)}"))
                    rows = res.fetchall()
                    cols = list(res.keys())
                    for r in rows:
                        samples.append(dict(zip(cols, r)))
                except Exception as e:
                    print(f"[get_db_schema] sampling {t} failed: {e}")

                tables_meta.append({
                    "schema": schema,
                    "table_name": t,
                    "columns": cols_meta,
                    "foreign_keys": fks,
                    "samples": samples,
                })
                count += 1

            if count >= max_tables:
                break

    return {"tables": tables_meta}


def pick_candidate_tables(question: str, schema_json: dict, topk: int = 3):
    q_tokens = set(re.findall(r"[A-Za-z_]+", question.lower()))
    scored = []
    for t in schema_json["tables"]:
        t_name = f"{t['table_name']}".strip(".").lower()
        t_tokens = set(re.findall(r"[A-Za-z_]+", t_name))
        col_tokens = set()
        for c in t.get("columns", []):
            col_tokens |= set(re.findall(r"[A-Za-z_]+", (c.get("name","") + " " + str(c.get("type",""))).lower()))
        score = len(q_tokens & (t_tokens | col_tokens))
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:topk] if _ > 0]


def build_sql_prompt(question: str, schema_json: dict):
    candidates = pick_candidate_tables(question, schema_json, topk=4)
    # Fallback: nimm alles, wenn kein Kandidat scored
    if not candidates:
        candidates = schema_json["tables"]

    def fmt_table(t):
        fq = f"{t['table_name']}"
        cols = ", ".join(c['name'] for c in t.get("columns", []))
        sample_lines = []
        for s in t.get("samples", [])[:2]:
            sample_lines.append(json.dumps(s, ensure_ascii=False))
        return (
                f"- table_name: {fq}\n"
                f"  columns: {cols}\n"
                f"  samples:\n    " + ("\n    ".join(sample_lines) if sample_lines else "(none)")
        )

    schema_block = "\n".join(fmt_table(t) for t in candidates)

    return f"""
You are an expert SQL generator. Generate a single **valid** SELECT statement for SQLite/SQLAlchemy.

Rules:
- Use only the tables and columns listed in SCHEMA below.
Hard rules:
- Output RAW SQL only – no markdown, no backticks, no code fences, no labels.
- Start with SELECT or WITH.
- Do not add comments or explanations.
- For SQLite, do NOT prefix with 'main.'; quote identifiers if needed with double quotes, and never quote dotted names as one token.

QUESTION:
{question}

SCHEMA (subset):
{schema_block}

Output ONLY the SQL:
""".strip()

