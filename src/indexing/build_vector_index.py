# Databricks notebook source
# MAGIC %md
# MAGIC # Build Vector Index for RAG
# MAGIC 
# MAGIC This notebook creates/updates a Delta Sync Vector Search index for the chunked documents.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Widgets for Wheel Installation
# MAGIC 
# MAGIC Define only widgets needed to construct wheel path before restart

# COMMAND ----------

# Define widgets needed for wheel installation
dbutils.widgets.text("catalog_name", "test_catalog", "Catalog Name")
dbutils.widgets.text("schema_name", "test_schema", "Schema Name")
dbutils.widgets.text("rag_utils_wheel", "rag_utils-0.1.0-py3-none-any.whl", "RAG Utils Wheel File")

# Get values needed for wheel path
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
rag_utils_wheel = dbutils.widgets.get("rag_utils_wheel")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC 
# MAGIC For serverless notebook tasks, dependencies must be installed via %pip

# COMMAND ----------

# Install PyPI dependencies first
# MAGIC %pip install --index-url https://pypi.org/simple databricks-vectorsearch

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install RAG Utils Wheel

# COMMAND ----------

# Install rag_utils wheel from Unity Catalog volume
# Use PyPI index explicitly to avoid Azure DevOps index issues
wheel_path = f"/Volumes/{catalog_name}/{schema_name}/libs/{rag_utils_wheel}"

# COMMAND ----------

# MAGIC %pip install --index-url https://pypi.org/simple --no-cache-dir $wheel_path
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Remaining Widgets and Get Parameters
# MAGIC 
# MAGIC After restart, define all widgets and get all values

# COMMAND ----------

# Define all widgets (including ones already defined, as restart cleared them)
dbutils.widgets.text("catalog_name", "test_catalog", "Catalog Name")
dbutils.widgets.text("schema_name", "test_schema", "Schema Name")
dbutils.widgets.text("volume_name", "test_volume", "Volume Name")
dbutils.widgets.text("chunked_table_name", "ctcbl_chunked_docs2", "Chunked Table Name")
dbutils.widgets.text("embedding_model", "gte-large-en-v1.5", "Embedding Model")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en-v1.5", "Embedding Endpoint Name")
dbutils.widgets.text("vector_index_name", "test_volume_ctcbl_chunked_index_element__v0_0_1", "Vector Index Name")
dbutils.widgets.text("vector_search_endpoint", "ctcbl-unstructured-endpoint", "Vector Search Endpoint Name")
dbutils.widgets.text("rag_utils_wheel", "rag_utils-0.1.0-py3-none-any.whl", "RAG Utils Wheel File")

# Get all values from widgets
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
volume_name = dbutils.widgets.get("volume_name")
chunked_table_name = dbutils.widgets.get("chunked_table_name")
embedding_model = dbutils.widgets.get("embedding_model")
embedding_endpoint = dbutils.widgets.get("embedding_endpoint")
vector_index_name = dbutils.widgets.get("vector_index_name")
vector_search_endpoint = dbutils.widgets.get("vector_search_endpoint")
rag_utils_wheel = dbutils.widgets.get("rag_utils_wheel")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

# Import from rag_utils package
from rag_utils.configs.databricks_config.utils import get_table_url
from rag_utils.embedding.index_builder import build_retriever_index, get_vector_index_row_count
from rag_utils.pipelines.data_pipeline_config import DataPipelineOutputConfig

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Output

# COMMAND ----------

# Construct full table names for DataPipelineOutputConfig
# Since we're using direct table names from widgets, construct the FQNs
chunked_docs_table_fqn = f"{catalog_name}.{schema_name}.{chunked_table_name}"
vector_index_fqn = f"{catalog_name}.{schema_name}.{vector_index_name}"
# Create a placeholder parsed_docs_table (not used for index building, but required by config)
# Use a pattern based on chunked_table_name, removing common chunked postfixes
base_name = chunked_table_name.replace("_chunked", "").replace("_docs", "")
parsed_docs_table_fqn = f"{catalog_name}.{schema_name}.{base_name}_docs"

# Create output configuration
output_config = DataPipelineOutputConfig(
    vector_search_endpoint=vector_search_endpoint,
    parsed_docs_table=parsed_docs_table_fqn,
    chunked_docs_table=chunked_docs_table_fqn,
    vector_index=vector_index_fqn,
)

# Validate catalog and schema
is_valid, msg = output_config.validate_catalog_and_schema()
if not is_valid:
    raise Exception(msg)

# Validate or create vector search endpoint
is_valid, msg = output_config.create_or_validate_vector_search_endpoint()
if not is_valid:
    raise Exception(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build/Update Vector Index

# COMMAND ----------

print(f"Building vector index: {output_config.vector_index}")
print(f"Source table: {output_config.chunked_docs_table}")
print(f"Embedding model: {embedding_model}")
print(f"Embedding endpoint: {embedding_endpoint}")
print(f"Vector search endpoint: {output_config.vector_search_endpoint}")

# Build or sync the vector index
# Note: build_retriever_index returns (is_error: bool, message: str)
is_error, message = build_retriever_index(
    vector_search_endpoint=output_config.vector_search_endpoint,
    chunked_docs_table_name=output_config.chunked_docs_table.replace("`",""),
    vector_search_index_name=output_config.vector_index,
    embedding_endpoint_name=embedding_endpoint,
    force_delete_index_before_create=False,  # Use sync instead of delete
    primary_key="chunk_id",
    embedding_source_column="chunk_content"
)

print(message)
if is_error:
    raise RuntimeError(message)

print("View VS index status:", get_table_url(output_config.vector_index))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Index

# COMMAND ----------

# Get and display index statistics
# Note: get_vector_index_row_count only needs the index name
index_row_count = get_vector_index_row_count(output_config.vector_index)
print(f"\nVector Index Statistics:")
print(f"  Index Name: {output_config.vector_index}")
print(f"  Row Count: {index_row_count}")
print(f"\nIndex created/updated successfully!")

