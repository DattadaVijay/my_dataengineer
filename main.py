import os
import json
import base64
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import PipelineLibrary, NotebookLibrary
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks import sql

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
def upload_pipeline_notebook(
    pipeline_name: str,
    source_type: str,
    source_catalog: str,
    source_schema: str,
    target_catalog: str,
    target_schema: str,
    target_table: str,
    key_columns: str,
    transformation_code: str,
    source_table: str = "",
    source_volume: str = "",
    file_format: str = "json",
    overwrite: bool = True
) -> str:
    """
    Generate and upload a DLT pipeline notebook to Databricks workspace.
    Call this first, then call create_dlt_pipeline with the returned notebook path.

    Args:
        pipeline_name: Name of the pipeline (e.g. azure_logs_pipeline)
        source_type: Either 'table' or 'volume'
        source_catalog: Source UC catalog
        source_schema: Source UC schema
        target_catalog: Target UC catalog
        target_schema: Target UC schema
        target_table: Target table name
        key_columns: Comma separated columns for deduplication (e.g. "time")
        transformation_code: Raw PySpark chained calls applied BEFORE dedup.
                             Must start with a dot. Example:
                             .withColumn("time", to_timestamp(col("time")))
                             Pass empty string if no transformations needed.
        source_table: Source table name (required if source_type is 'table')
        source_volume: Source volume name (required if source_type is 'volume')
        file_format: File format json/csv/parquet (required if source_type is 'volume')
        overwrite: Whether to overwrite existing notebook (default True)
    """
    w = get_client()

    key_cols_list = [f'"{k.strip()}"' for k in key_columns.split(",")]
    key_cols_str = ", ".join(key_cols_list)

    if source_type == "table":
        source_read = (
            f'spark.readStream.table("{source_catalog}.{source_schema}.{source_table}")'
        )
        # For table sources order by first key column — _metadata not available
        window_order = f'col("{key_columns.split(",")[0].strip()}")'
    else:
        source_read = (
            f'spark.readStream\n'
            f'            .format("cloudFiles")\n'
            f'            .option("cloudFiles.format", "{file_format}")\n'
            f'            .option("cloudFiles.schemaLocation",\n'
            f'                    "/Volumes/{target_catalog}/{target_schema}/_schema/{pipeline_name}")\n'
            f'            .load("/Volumes/{source_catalog}/{source_schema}/{source_volume}")'
        )
        # For volume sources _metadata.file_modification_time is available
        window_order = 'col("_metadata.file_modification_time")'

    # transformation_code passed verbatim — zero parsing
    transform_code = f"\n            {transformation_code.strip()}" if transformation_code.strip() else ""

    notebook_code = f'''import dlt
from pyspark.sql.functions import col, row_number, to_timestamp
from pyspark.sql.window import Window


@dlt.table(
    name="{target_table}_raw",
    comment="Raw streaming data from {source_type} source"
)
def {target_table}_raw():
    return (
        {source_read}
    )


@dlt.table(
    name="{target_table}",
    comment="Cleaned and deduplicated table"
)
def {target_table}():
    key_cols = [{key_cols_str}]
    window = Window.partitionBy(*key_cols).orderBy({window_order}.desc())
    return (
        dlt.read_stream("{target_table}_raw"){transform_code}
            .withColumn("_rank", row_number().over(window))
            .filter(col("_rank") == 1)
            .drop("_rank")
    )
'''

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
    Create a DLT pipeline from an already uploaded notebook.
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

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = port
    mcp.settings.stateless_http = True
    mcp.settings.transport_security = None
    mcp.run(transport="streamable-http")