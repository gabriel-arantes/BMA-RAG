from dlt import *
from typing import Dict, Any, List, Optional # not used?
from pyspark.sql.functions import expr, explode, parse_json, col, to_json, regexp_extract, current_timestamp, md5
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType # not used?
import logging # not used?
import json # not used?

# Import rag_utils (from pipeline libraries)
from rag_utils.chunking.strategies import get_elements_chunking_udf
from rag_utils.chunking.options import ElementsChunkerOptions

# Get configuration from pipeline settings (with defaults)
catalog_name = spark.conf.get("bundle.catalog_name", "test_catalog")
schema_name = spark.conf.get("bundle.schema_name", "test_schema")
volume_name = spark.conf.get("bundle.volume_name", "test_volume")

# Get remaining configuration from pipeline settings (with defaults)
source_table_name = spark.conf.get("bundle.source_table_name", "ctcbl_raw_files2")
parsed_table_name = spark.conf.get("bundle.parsed_table_name", "ctcbl_ai_parse2")
chunked_table_name = spark.conf.get("bundle.chunked_table_name", "ctcbl_chunked_docs2")
source_volume_path = spark.conf.get("bundle.source_volume_path", f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/rag_source_files/")
parsed_volume_path = spark.conf.get("bundle.parsed_volume_path", f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/parsed_dir/")
schema_location = spark.conf.get("bundle.schema_location", f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/parsed_dir/schema")

# Configure chunker options from pipeline settings
chunker_options = ElementsChunkerOptions(
    max_chunk_size=int(spark.conf.get("bundle.max_chunk_size", "4000")),
    min_chunk_size=int(spark.conf.get("bundle.min_chunk_size", "400")),
    embedding_model=spark.conf.get("bundle.embedding_model", "gte-large-en-v1.5"),
    config_path=spark.conf.get("bundle.config_path", "/Workspace/Users/garantes_oc@bma.bm/config/rag_config.json")
)

# Create the chunking UDF
db_chunking_udf = get_elements_chunking_udf(chunker_options)


# LOAD RAW DOCUMENTS
@dlt.table(
    name=source_table_name,
    comment="Raw files from UC Volume",
    table_properties={
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true",
        "quality": "bronze"
    }
)
def raw_files():
  return (
    spark.readStream.format("cloudFiles") \
    .option("cloudFiles.format", "binaryFile") \
    .option("cloudFiles.schemaLocation", schema_location) \
    .option("cloudFiles.inferColumnTypes", "true") \
    .option("recursiveFiles", "true") \
    .option("pathGlobFilter", "*.pdf") \
    .load(source_volume_path) \
    .withColumn("processing_timestamp", current_timestamp()) 
    )

#PARSE DOCUMENTS
@dlt.table(
    name=parsed_table_name,
    comment="Parsed documents for retrieval",
    table_properties={
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true",
        "delta.feature.variantType-preview" : "supported",
        "quality": "silver"
    }
)


### ai_parse_document version 2 is implemented in DBR 17, but DLT is in 16.4
### we can switch once it's available
### Adjust to use the version parameter inside "map" so it's easier to switch
@dlt.expect("valid_content", "elements IS NOT NULL")
@dlt.expect("valid_path", "path IS NOT NULL")
def parsed_docs():
    return (
        dlt.read(source_table_name).withColumn(
        "parsed_doc",
        expr(f"""
            ai_parse_document(
                content, 
                map(
                    'imageOutputPath', '{parsed_volume_path}',
                    'descriptionElementTypes', '*'
                )
            )
        """)
    )
    .withColumn(
        "parsed_json",
        parse_json(col("parsed_doc").cast("string"))
    )
    .withColumn("document_name", regexp_extract(col("path"), '([^/]+)$', 1))
    .withColumn("processed_at", current_timestamp())
    .selectExpr(
        "path",
        "document_name",
        "processed_at",
        "parsed_json:document:elements",
        "parsed_json:error_status",
        "parsed_json:metadata"
    )
)

# CHUNK DOCUMENT CONTENTS FOR VECTOR DATABASE
@dlt.table(
    name=chunked_table_name,
    comment="Chunked document contents for vector search",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.zOrderCols": "chunk_id",
        "pipelines.autoOptimize.optimizeWrite": "true",
        "delta.enableChangeDataFeed" : "true"
    }
)
@dlt.expect("valid_chunk", "chunk_content IS NOT NULL")
@dlt.expect("valid_chunk_id", "chunk_id IS NOT NULL")
def chunked_docs():
    df_with_json = dlt.read(parsed_table_name).withColumn("elements_json", to_json(col("elements")))
    return (
        df_with_json.withColumn("chunks",db_chunking_udf(col("elements_json"))).select(
        "*",
        explode(col("chunks")).alias("chunk_data")
    ).select(
        col("path"),
        col("document_name").alias("file_name"),
        col("processed_at"),
        col("chunk_data.content").alias("chunk_content"),
        col("chunk_data.section").alias("chunk_section"),
        col("chunk_data.chunk_type").alias("chunk_type")
    ).withColumn("chunk_id", md5("chunk_content"))
    )