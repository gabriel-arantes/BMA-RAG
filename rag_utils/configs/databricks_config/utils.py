"""
configs/databricks_config/utils.py

Minimal utilities for Databricks workspace operations.
Extracted to be self-contained.
"""
import os
import json
import subprocess
from typing import Optional


def get_databricks_cli_config() -> Optional[dict]:
    """
    Retrieve the Databricks CLI configuration by running 'databricks auth describe' command.

    Returns:
        dict: The parsed JSON configuration from the Databricks CLI, or None if an error occurs

    Note:
        Requires the Databricks CLI to be installed and configured
    """
    try:
        # Run databricks auth describe command and capture output
        process = subprocess.run(
            ["databricks", "auth", "describe", "-o", "json"],
            capture_output=True,
            text=True,
            check=True,  # Raises CalledProcessError if command fails
        )

        # Parse JSON output
        return json.loads(process.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, Exception):
        return None


def get_workspace_hostname() -> str:
    """
    Return the Databricks workspace base URL (e.g., https://adb-12345.azuredatabricks.net).

    Priority:
      1) Databricks runtime context (works in notebooks AND jobs)
      2) Env vars: DATABRICKS_HOST / DATABRICKS_URL
      3) Databricks CLI config (~/.databrickscfg)
    """
    # 1) Databricks runtime context (covers notebooks and jobs)
    try:
        # Try the newer context path first
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        # Try the Python DBUtils shim (present on most DBRs)
        try:
            from pyspark.dbutils import DBUtils
            dbutils = DBUtils(spark)
            ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
            if ctx.apiUrl().isDefined():
                return ctx.apiUrl().get().rstrip("/")
            if ctx.browserHostName().isDefined():
                return f"https://{ctx.browserHostName().get()}".rstrip("/")
        except Exception:
            pass

        # Fallback via JVM API (older DBR compatibility)
        try:
            jctx = spark.sparkContext._jvm.com.databricks.dbutils_v1.DBUtilsV1().notebook().getContext()
            if jctx.apiUrl().isDefined():
                return jctx.apiUrl().get().rstrip("/")
            if jctx.browserHostName().isDefined():
                return f"https://{jctx.browserHostName().get()}".rstrip("/")
        except Exception:
            pass
    except Exception:
        pass

    # 2) Environment variables (useful locally or if set on jobs/clusters)
    for key in ("DATABRICKS_HOST", "DATABRICKS_URL"):
        val = os.environ.get(key)
        if val:
            return val.rstrip("/")

    # 3) CLI config (~/.databrickscfg) – your existing helper
    cli_config = get_databricks_cli_config()
    if cli_config and cli_config.get("details", {}).get("host"):
        return cli_config["details"]["host"].rstrip("/")

    raise RuntimeError(
        "Could not determine Databricks workspace host. "
        "Set DATABRICKS_HOST (or DATABRICKS_URL) or ensure the runtime context/CLI config is available."
    )


def get_table_url(table_fqdn: str) -> str:
    """
    Generate the URL for a Unity Catalog table in the Databricks UI.

    Args:
        table_fqdn: Fully qualified table name in format 'catalog.schema.table'.
                   Can optionally include backticks around identifiers.

    Returns:
        str: The full URL to view the table in the Databricks UI.

    Example:
        >>> get_table_url("main.default.my_table")
        'https://my-workspace.cloud.databricks.com/explore/data/main/default/my_table'
    """
    table_fqdn = table_fqdn.replace("`", "")
    catalog, schema, table = table_fqdn.split(".")
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/explore/data/{catalog}/{schema}/{table}"
    return url

