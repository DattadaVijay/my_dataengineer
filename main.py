import os
import json
import base64
import time 
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import PipelineLibrary, NotebookLibrary
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks import sql
from databricks.sdk.service.compute import ClusterSpec 
from databricks.sdk.service.jobs import SubmitTask, NotebookTask    

load_dotenv()

mcp = FastMCP("databricks-context")

def get_client():
    return WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"]
    )

def get_sql_connection():
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"]
    )

def run_query(query: str) -> list[dict]:
    with get_sql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

# ── Layer 1 — UC Metadata ─────────────────────────────────────────────────────

@mcp.tool()
def get_schemas(catalog: str) -> str:
    """List all schemas in a Unity Catalog catalog."""
    w = get_client()
    schemas = list(w.schemas.list(catalog_name=catalog))
    if not schemas:
        return f"No schemas found in catalog '{catalog}'"
    return "\n".join([s.name for s in schemas])

@mcp.tool()
def get_tables(catalog: str, schema: str) -> str:
    """List all tables in a Unity Catalog schema."""
    w = get_client()
    tables = list(w.tables.list(catalog_name=catalog, schema_name=schema))
    if not tables:
        return f"No tables found in {catalog}.{schema}"
    return "\n".join([t.name for t in tables])

@mcp.tool()
def get_table_schema(catalog: str, schema: str, table: str) -> str:
    """Get column names and types for a Unity Catalog table."""
    w = get_client()
    full_name = f"{catalog}.{schema}.{table}"
    t = w.tables.get(full_name=full_name)
    if not t.columns:
        return f"No columns found for {full_name}"
    return "\n".join([f"{col.name}: {col.type_text}" for col in t.columns])

@mcp.tool()
def get_volume_contents(catalog: str, schema: str, volume: str) -> str:
    """List files inside a Unity Catalog volume."""
    w = get_client()
    path = f"/Volumes/{catalog}/{schema}/{volume}"
    files = list(w.files.list_directory_contents(directory_path=path))
    if not files:
        return f"Volume '{path}' is empty."
    return "\n".join([f.path for f in files])

# ── Layer 2 — Data Quality + Volume Intelligence ──────────────────────────────

@mcp.tool()
def get_table_row_count(catalog: str, schema: str, table: str) -> str:
    """Get the total row count of a table."""
    rows = run_query(f"SELECT COUNT(*) as count FROM {catalog}.{schema}.{table}")
    return f"Row count: {rows[0]['count']}"

@mcp.tool()
def get_null_counts(catalog: str, schema: str, table: str) -> str:
    """Get null percentage for each column in a table."""
    full = f"{catalog}.{schema}.{table}"
    w = get_client()
    t = w.tables.get(full_name=full)
    cols = [c.name for c in t.columns]
    null_exprs = ", ".join([
        f"ROUND(SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS {c}"
        for c in cols
    ])
    rows = run_query(f"SELECT {null_exprs} FROM {full}")
    if not rows:
        return "No data."
    return "\n".join([f"{k}: {v}% null" for k, v in rows[0].items()])

@mcp.tool()
def get_duplicate_rate(catalog: str, schema: str, table: str, key_columns: str) -> str:
    """Check duplicate rate for given key columns (comma separated)."""
    full = f"{catalog}.{schema}.{table}"
    keys = ", ".join([k.strip() for k in key_columns.split(",")])
    rows = run_query(f"""
        SELECT COUNT(*) as total,
               COUNT(DISTINCT {keys}) as unique_keys,
               ROUND((COUNT(*) - COUNT(DISTINCT {keys})) * 100.0 / COUNT(*), 2) as dup_pct
        FROM {full}
    """)
    r = rows[0]
    return f"Total: {r['total']} | Unique: {r['unique_keys']} | Duplicate rate: {r['dup_pct']}%"

@mcp.tool()
def get_data_skew(catalog: str, schema: str, table: str, partition_column: str) -> str:
    """Check data distribution across a partition column."""
    full = f"{catalog}.{schema}.{table}"
    rows = run_query(f"""
        SELECT {partition_column}, COUNT(*) as row_count
        FROM {full}
        GROUP BY {partition_column}
        ORDER BY row_count DESC
        LIMIT 20
    """)
    if not rows:
        return "No data."
    return "\n".join([f"{r[partition_column]}: {r['row_count']} rows" for r in rows])

@mcp.tool()
def get_table_sample(catalog: str, schema: str, table: str) -> str:
    """Get 5 sample rows from a table to understand structure."""
    rows = run_query(f"SELECT * FROM {catalog}.{schema}.{table} LIMIT 5")
    if not rows:
        return "No data."
    return "\n".join([str(r) for r in rows])

@mcp.tool()
def get_volume_file_schema(catalog: str, schema: str, volume: str, file_format: str = "json") -> str:
    """
    Infer schema from files in a UC volume by sampling the first file.
    Supported formats: json, csv, parquet.
    """
    rows = run_query(f"""
        SELECT *
        FROM read_files(
            '/Volumes/{catalog}/{schema}/{volume}',
            format => '{file_format}'
        )
        LIMIT 1
    """)
    if not rows:
        return "Could not infer schema — volume may be empty or unsupported format."
    cols = list(rows[0].keys())
    return f"Inferred columns: {', '.join(cols)}\nSample row: {json.dumps(rows[0], default=str)}"

# ── Layer 3 — Pipeline Management ────────────────────────────────────────────

@mcp.tool()
def list_pipelines() -> str:
    """List all DLT pipelines in the workspace."""
    w = get_client()
    pipelines = list(w.pipelines.list_pipelines())
    if not pipelines:
        return "No pipelines found."
    return "\n".join([f"{p.pipeline_id}: {p.name}" for p in pipelines])

@mcp.tool()
def get_pipeline_status(pipeline_id: str) -> str:
    """Get the current status of a DLT pipeline."""
    w = get_client()
    p = w.pipelines.get(pipeline_id=pipeline_id)
    return f"Name: {p.name}\nState: {p.state}\nHealth: {p.health}"

@mcp.tool()
def get_pipeline_by_name(pipeline_name: str) -> str:
    """
    Check if a DLT pipeline with the given name already exists.
    Returns pipeline_id if FOUND, or NOT_FOUND if it does not exist.
    Always call this before deploying.
    If FOUND → call update_dlt_pipeline.
    If NOT_FOUND → call create_dlt_pipeline.

    Args:
        pipeline_name: Exact name of the pipeline to look up
    """
    w = get_client()
    pipelines = list(w.pipelines.list_pipelines())
    for p in pipelines:
        if p.name == pipeline_name:
            return (
                f"FOUND\n"
                f"pipeline_id: {p.pipeline_id}\n"
                f"name: {p.name}\n"
                f"state: {p.state}"
            )
    return "NOT_FOUND"

@mcp.tool()
def upload_pipeline_notebook(
    pipeline_name: str,
    notebook_code: str,
    overwrite: bool = True
) -> str:
    """
    Upload a DLT pipeline notebook to Databricks workspace at /Shared/pipelines/{pipeline_name}.

    YOU write the complete notebook_code as valid Python — this tool only uploads it verbatim.
    No templates, no code generation, no modifications to your code whatsoever.
    Whatever you pass is exactly what gets uploaded.

    Always call get_table_schema and get_table_sample first to understand the data,
    then write the full correct DLT notebook yourself and pass it here.

    Args:
        pipeline_name: Name used for the notebook path (e.g. azure_logs_pipeline)
        notebook_code: Complete valid Python DLT notebook code written by you.
        overwrite: Whether to overwrite existing notebook (default True)

    Example notebook_code for streaming from a UC table with two dependent tables:
        import dlt
        from pyspark.sql.functions import col, to_timestamp, date_trunc, count

        @dlt.table(name="azure_logs_clean")
        def azure_logs_clean():
            return (
                spark.readStream
                    .table("dltvijay.source.azure_logs")
                    .withColumn("time", to_timestamp(col("time")))
            )

        @dlt.table(name="azure_logs_report")
        def azure_logs_report():
            return (
                dlt.read("azure_logs_clean")
                    .groupBy(date_trunc("hour", col("time")).alias("hour"))
                    .agg(count("*").alias("log_count"))
            )
    """
    w = get_client()

    notebook_path = f"/Shared/pipelines/{pipeline_name}"

    try:
        w.workspace.mkdirs(path="/Shared/pipelines")
    except Exception:
        pass

    w.workspace.import_(
        path=notebook_path,
        content=base64.b64encode(notebook_code.encode()).decode(),
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        overwrite=overwrite
    )

    return f"Notebook uploaded successfully to: {notebook_path}"

@mcp.tool()
def create_dlt_pipeline(
    pipeline_name: str,
    notebook_path: str,
    target_catalog: str,
    target_schema: str,
    allow_duplicate_names: bool = False
) -> str:
    """
    Create a new DLT pipeline from an already uploaded notebook.
    Only call this when get_pipeline_by_name returns NOT_FOUND.
    Always call upload_pipeline_notebook first to get the notebook_path.

    Args:
        pipeline_name: Name of the pipeline
        notebook_path: Workspace path returned by upload_pipeline_notebook
        target_catalog: Target UC catalog
        target_schema: Target UC schema
        allow_duplicate_names: Set True to allow creating pipeline with duplicate name
    """
    w = get_client()

    pipeline = w.pipelines.create(
        name=pipeline_name,
        catalog=target_catalog,
        schema=target_schema,
        libraries=[
            PipelineLibrary(
                notebook=NotebookLibrary(path=notebook_path)
            )
        ],
        continuous=False,
        development=True,
        serverless=True,
        allow_duplicate_names=allow_duplicate_names
    )

    return (
        f"Pipeline '{pipeline_name}' created successfully.\n"
        f"Pipeline ID: {pipeline.pipeline_id}\n"
        f"Notebook: {notebook_path}\n"
        f"Target: {target_catalog}.{target_schema}"
    )

@mcp.tool()
def update_dlt_pipeline(
    pipeline_id: str,
    pipeline_name: str,
    notebook_path: str,
    target_catalog: str,
    target_schema: str
) -> str:
    """
    Update an existing DLT pipeline with a newly uploaded notebook.
    Only call this when get_pipeline_by_name returns FOUND.
    Always call upload_pipeline_notebook first to upload the updated code.

    Args:
        pipeline_id: Pipeline ID returned by get_pipeline_by_name
        pipeline_name: Name of the pipeline (same as existing)
        notebook_path: Workspace path returned by upload_pipeline_notebook
        target_catalog: Target UC catalog
        target_schema: Target UC schema
    """
    w = get_client()

    w.pipelines.update(
        pipeline_id=pipeline_id,
        name=pipeline_name,
        catalog=target_catalog,
        schema=target_schema,
        libraries=[
            PipelineLibrary(
                notebook=NotebookLibrary(path=notebook_path)
            )
        ],
        continuous=False,
        development=True,
        serverless=True
    )

    return (
        f"Pipeline '{pipeline_name}' updated successfully.\n"
        f"Pipeline ID: {pipeline_id}\n"
        f"Notebook: {notebook_path}\n"
        f"Target: {target_catalog}.{target_schema}"
    )


# ── Layer 4 — App Deployment + Playwright Testing ─────────────────────────────

@mcp.tool()
def create_databricks_app(
    app_name: str,
    app_code: str,
) -> str:
    """
    Create a Databricks App, upload Streamlit source code, and deploy.
    Waits for app to be ready then deploys automatically.
    After this succeeds, call get_app_url() to get the live URL.

    IMPORTANT: app_code MUST be valid Streamlit Python code.
    Always use streamlit as the framework — no Flask, no Gradio, no plain Python.
    The entry file must be app.py — Databricks Apps requires this exact filename.

    A minimal valid Streamlit app looks like:
        import streamlit as st
        st.title("Hello World")
        st.write("App is running.")

    Args:
        app_name: Name of the app (used for workspace path and app registry)
        app_code: Complete valid Streamlit Python code
    """
    from databricks.sdk.service.apps import App, AppDeployment, AppDeploymentMode
    import traceback

    w = get_client()
    user = os.environ["DATABRICKS_USER"]
    workspace_path = f"/Users/{user}/{app_name}"
    file_path = f"{workspace_path}/app.py"
    app_path = f"/Workspace/Users/{user}/{app_name}"

    try:
        # Step 1 — upload app.py to workspace
        try:
            w.workspace.mkdirs(path=workspace_path)
        except Exception:
            pass

        w.workspace.import_(
            path=file_path,
            content=base64.b64encode(app_code.encode()).decode(),
            format=ImportFormat.SOURCE,
            language=Language.PYTHON,
            overwrite=True
        )
        print(f"SUCCESS: uploaded to {file_path}")

        # Step 2 — create app if needed, wait until ready
        existing_names = [a.name for a in w.apps.list()]
        print(f"INFO: existing apps = {existing_names}")

        if app_name not in existing_names:
            print(f"INFO: creating app {app_name} and waiting for ready state...")
            w.apps.create_and_wait(App(name=app_name))
            print(f"SUCCESS: app ready")
        else:
            print(f"INFO: app already exists")

        # Step 3 — deploy source code and wait
        print(f"INFO: deploying source from {app_path}")
        w.apps.deploy_and_wait(
            app_name=app_name,
            app_deployment=AppDeployment(
                source_code_path=app_path,
                mode=AppDeploymentMode.SNAPSHOT,
            ),
        )
        print(f"SUCCESS: deployment complete")

        return (
            f"App '{app_name}' deployed successfully.\n"
            f"Source: {file_path}\n"
            f"Now call get_app_url('{app_name}') to get the live URL."
        )

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"ERROR in create_databricks_app:\n{error_msg}")
        return f"Deployment failed with error:\n{str(e)}\n\nFull traceback:\n{error_msg}"


@mcp.tool()
def get_app_url(app_name: str, max_wait_seconds: int = 120) -> str:
    """
    Poll until the Databricks App is RUNNING or ACTIVE and return its URL.
    Call this after create_databricks_app() succeeds.
    Times out after max_wait_seconds (default 120).

    Args:
        app_name: Name of the deployed app
        max_wait_seconds: How long to wait before giving up
    """
    w = get_client()
    waited = 0
    poll_interval = 5
    state = "UNKNOWN"

    while waited < max_wait_seconds:
        app = w.apps.get(name=app_name)
        state = str(app.compute_status.state) if app.compute_status else "UNKNOWN"

        if any(s in state.upper() for s in ["RUNNING", "ACTIVE"]) and app.url:
            return (
                f"App '{app_name}' is {state}.\n"
                f"URL: {app.url}\n"
                f"Pass this URL to upload_playwright_test_notebook()"
            )

        if any(s in state.upper() for s in ["ERROR", "FAILED", "STOPPED"]):
            return f"App '{app_name}' entered state: {state}. Check deployment logs."

        time.sleep(poll_interval)
        waited += poll_interval

    return f"Timed out after {max_wait_seconds}s. Last state: {state}"


@mcp.tool()
def upload_playwright_test_notebook(
    app_name: str,
    app_url: str,
    test_code: str,
) -> str:
    """
    Upload a Playwright UI test notebook targeting the deployed app URL.
    YOU write the complete test_code — this tool only uploads it verbatim.

    The notebook runs inside Databricks so it has workspace auth natively.
    Use the DATABRICKS_TOKEN env var for the Authorization header.

    Args:
        app_name: Used to name the notebook path
        app_url: The live app URL returned by get_app_url()
        test_code: Complete Python Playwright test code you have written
    """
    w = get_client()
    user = os.environ["DATABRICKS_USER"]
    notebook_path = f"/Users/{user}/{app_name}/tests"

    try:
        w.workspace.mkdirs(path=notebook_path)
    except Exception:
        pass

    full_path = f"{notebook_path}/ui_test"
    w.workspace.import_(
        path=full_path,
        content=base64.b64encode(test_code.encode()).decode(),
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        overwrite=True
    )

    return (
        f"Test notebook uploaded to: {full_path}\n"
        f"Targeting app URL: {app_url}\n"
        f"Call run_and_get_test_results('{full_path}') to execute."
    )


@mcp.tool()
def run_and_get_test_results(notebook_path: str, timeout_seconds: int = 300) -> str:
    """
    Run the Playwright test notebook as a one-time job and return results.
    Polls until completion then returns stdout output.

    Args:
        notebook_path: Workspace path to the test notebook
        timeout_seconds: Max wait time (default 300s)
    """
    w = get_client()
    life_cycle = "UNKNOWN"

    run = w.jobs.submit(
        run_name=f"playwright-test-{int(time.time())}",
        tasks=[
            SubmitTask(
                task_key="ui_test",
                notebook_task=NotebookTask(
                    notebook_path=notebook_path,
                ),
                new_cluster=ClusterSpec(
                    spark_version="15.4.x-scala2.12",
                    node_type_id="Standard_DS3_v2",
                    num_workers=0,
                    spark_conf={"spark.master": "local[*]"},
                )
            )
        ]
    )

    run_id = run.run_id
    waited = 0
    poll_interval = 10

    while waited < timeout_seconds:
        run_state = w.jobs.get_run(run_id=run_id)
        life_cycle = str(run_state.state.life_cycle_state)

        if "TERMINATED" in life_cycle:
            result_state = str(run_state.state.result_state)
            output = w.jobs.get_run_output(run_id=run_id)
            notebook_output = output.notebook_output.result if output.notebook_output else ""
            logs = output.logs or ""

            return (
                f"Run {run_id} finished: {result_state}\n\n"
                f"--- Output ---\n{notebook_output}\n"
                f"--- Logs ---\n{logs[:2000]}"
            )

        time.sleep(poll_interval)
        waited += poll_interval

    return f"Timed out after {timeout_seconds}s. Run ID: {run_id} still in state: {life_cycle}"

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = port
    mcp.settings.stateless_http = True
    mcp.settings.transport_security = None
    mcp.run(transport="streamable-http")