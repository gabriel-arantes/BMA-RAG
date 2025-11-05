"""
embedding/embedding_config.py

Minimal tokenizer utilities for element-based chunking.
Extracted from the main rag_utils to be self-contained.
"""
from typing import Callable, Optional

# Hugging Face cache directory
HF_CACHE_DIR = "/Volumes/test_catalog/test_schema/test_volume/transformers_cache/hf_cache/"


def get_embedding_model_tokenizer(embedding_model: str, cache_dir: Optional[str] = None) -> Optional[Callable]:
    """
    Get tokenizer factory function for the specified embedding model.
    
    Args:
        embedding_model: Model name (e.g., "gte-large-en-v1.5")
        cache_dir: Optional cache directory (defaults to HF_CACHE_DIR)
        
    Returns:
        Callable that returns the tokenizer when called, or None if model not supported
    """
    from transformers import AutoTokenizer
    import tiktoken

    if cache_dir is None:
        cache_dir = HF_CACHE_DIR

    # Minimal model definitions - only what's needed for element chunking
    # Can be extended if more models are needed
    EMBEDDING_MODELS_W_TOKENIZER = {
        "gte-large-en-v1.5": {
            "tokenizer": lambda: AutoTokenizer.from_pretrained(
                "Alibaba-NLP/gte-large-en-v1.5", cache_dir=cache_dir
            ),
        },
        "bge-large-en-v1.5": {
            "tokenizer": lambda: AutoTokenizer.from_pretrained(
                "BAAI/bge-large-en-v1.5", cache_dir=cache_dir
            ),
        },
        "bge_large_en_v1_5": {
            "tokenizer": lambda: AutoTokenizer.from_pretrained(
                "BAAI/bge-large-en-v1.5", cache_dir=cache_dir
            ),
        },
        "text-embedding-ada-002": {
            "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-ada-002"),
        },
        "text-embedding-3-small": {
            "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-small"),
        },
        "text-embedding-3-large": {
            "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-large"),
        },
    }
    
    model_config = EMBEDDING_MODELS_W_TOKENIZER.get(embedding_model)
    if model_config:
        return model_config.get("tokenizer")
    return None

