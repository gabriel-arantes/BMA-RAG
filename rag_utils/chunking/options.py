"""
chunking/options.py

Pydantic options model for ElementsChunker configuration.
"""
from pydantic import BaseModel, Field


class ElementsChunkerOptions(BaseModel):
    """Configuration options for ElementsChunker."""
    
    max_chunk_size: int = Field(
        default=4000,
        description="Maximum chunk size in tokens"
    )
    
    min_chunk_size: int = Field(
        default=400,
        description="Minimum chunk size in tokens"
    )
    
    embedding_model: str = Field(
        default="gte-large-en-v1.5",
        description="Embedding model name for tokenizer"
    )
    
    config_path: str = Field(
        default="",
        description="Optional path to JSON config file for backwards compatibility"
    )
    
    cache_dir: str = Field(
        default="",
        description="Optional cache directory for tokenizers (defaults to HF_CACHE_DIR from embedding_config)"
    )

