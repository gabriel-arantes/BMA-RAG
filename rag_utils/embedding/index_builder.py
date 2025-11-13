"""
embedding/index_builder.py

Utilities for building and managing Databricks Vector Search indexes.
"""
from __future__ import annotations
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist, BadRequest
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)

# optional (nice for printed links in notebooks)
try:
    from ..configs.databricks_config.utils import get_table_url
except Exception:
    def get_table_url(x: str) -> str:
        return x


def build_retriever_index(
    *,
    vector_search_endpoint: str,
    chunked_docs_table_name: str,
    vector_search_index_name: str,
    embedding_endpoint_name: str,
    force_delete_index_before_create: bool = True,
    primary_key: str = "element_id",
    embedding_source_column: str = "text",
) -> tuple[bool, str]:
    """
    Create/sync a Databricks Vector Search Delta Sync index on your chunked docs table.

    Args:
        vector_search_endpoint: Name of the vector search endpoint
        chunked_docs_table_name: Fully qualified name of the chunked docs table
        vector_search_index_name: Name for the vector search index
        embedding_endpoint_name: Name of the embedding endpoint to use
        force_delete_index_before_create: If True, delete existing index before creating new one
        primary_key: Primary key column name (default: "element_id")
        embedding_source_column: Column name containing text to embed (default: "text")

    Returns:
        (is_error: bool, message: str)
    """
    w = WorkspaceClient()
    vsc = w.vector_search_indexes

    def find_index(index_name):
        try:
            return vsc.get_index(index_name=index_name)
        except ResourceDoesNotExist:
            return None

    def wait_for_index_to_be_ready(index):
        while index and not index.status.ready:
            print(f"Index {vector_search_index_name} exists but is not ready; waiting 30s…")
            time.sleep(30)
            index = find_index(index_name=vector_search_index_name)

    def wait_for_index_to_be_deleted(index):
        while index:
            print(f"Waiting for index {vector_search_index_name} to be deleted; waiting 30s…")
            time.sleep(30)
            index = find_index(index_name=vector_search_index_name)

    existing_index = find_index(index_name=vector_search_index_name)

    if existing_index:
        print(f"Found existing index: {get_table_url(vector_search_index_name)}")
        if force_delete_index_before_create:
            print(f"Deleting index {vector_search_index_name}…")
            vsc.delete_index(index_name=vector_search_index_name)
            wait_for_index_to_be_deleted(existing_index)
            create_index = True
        else:
            wait_for_index_to_be_ready(existing_index)
            create_index = False
            print("Starting an index sync (Delta Sync)…")
            try:
                vsc.sync_index(index_name=vector_search_index_name)
                return (False, f"Kicked off index sync for {vector_search_index_name}.")
            except BadRequest:
                return (True, "Index sync already in progress. Try again after it finishes.")
    else:
        print(f'Creating new vector search index "{vector_search_index_name}" on endpoint "{vector_search_endpoint}"')
        create_index = True

    if create_index:
        print("Creating Delta Sync index (computing embeddings may take a while)…")
        try:
            delta_sync_spec = DeltaSyncVectorIndexSpecRequest(
                source_table=chunked_docs_table_name,
                pipeline_type=PipelineType.TRIGGERED,
                embedding_source_columns=[
                    EmbeddingSourceColumn(
                        name=embedding_source_column,
                        embedding_model_endpoint_name=embedding_endpoint_name,
                    )
                ],
            )
            vsc.create_index(
                name=vector_search_index_name,
                endpoint_name=vector_search_endpoint,
                primary_key=primary_key,
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec=delta_sync_spec,
            )
            msg = f"Successfully created vector search index {vector_search_index_name}."
            return (False, msg)
        except Exception as e:
            return (True, f"Vector search index creation failed: {e}")


def get_vector_index_row_count(vector_search_index_name: str) -> int | None:
    """
    Get the number of indexed rows for a vector search index.

    Args:
        vector_search_index_name: Name of the vector search index

    Returns:
        Number of indexed rows, or None if unavailable
    """
    w = WorkspaceClient()
    vsc = w.vector_search_indexes
    try:
        index = vsc.get_index(index_name=vector_search_index_name)
        # status object can vary; just print to help you debug
        print(f"VectorIndexStatus attributes: {vars(index.status)}")
        return getattr(index.status, "indexed_row_count", None)
    except Exception as e:
        print(f"Error retrieving row count for {vector_search_index_name}: {e}")
        return None

