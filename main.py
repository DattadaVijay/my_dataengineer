import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from databricks.sdk import WorkspaceClient

load_dotenv()

mcp = FastMCP("databricks-context")

def get_client():
    return WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"]
    )

@mcp.tool()
def get_schemas(catalog: str) -> str:
    """List all schemas in a Unity Catalog catalog."""
    w = get_client()
    schemas = list(w.schemas.list(catalog_name=catalog))
    if not schemas:
        return f"No schemas found in catalog '{catalog}'"
    return "\n".join([s.name for s in schemas])

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = port
    mcp.settings.stateless_http = True
    mcp.settings.allowed_hosts = ["*"]
    mcp.run(transport="streamable-http")