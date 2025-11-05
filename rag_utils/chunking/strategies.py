"""
chunking/strategies.py

ElementsChunker class and Spark UDF helper for element-based chunking.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging
import json

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

from .options import ElementsChunkerOptions
from ..embedding.embedding_config import (
    get_embedding_model_tokenizer,
    HF_CACHE_DIR
)


class ElementsChunker:
    """
    A simplified chunking strategy that uses structured elements data
    to create semantically meaningful chunks based on section headers.
    """
    
    def __init__(self, options: ElementsChunkerOptions):
        """
        Initialize ElementsChunker with options.
        
        Args:
            options: ElementsChunkerOptions configuration object
        """
        self.max_chunk_size = options.max_chunk_size
        self.min_chunk_size = options.min_chunk_size
        self.embedding_model = options.embedding_model
        self.config_path = options.config_path
        self.cache_dir = options.cache_dir if options.cache_dir else HF_CACHE_DIR
        
        # Load configuration (optional, for backwards compatibility)
        self.config = self._load_config()

        # Content types that should be treated as semantic atomic units
        self.atomic_types = {'table', 'figure'}
        
        # Initialize tokenizer lazily
        self._tokenizer = None
        self._tokenizer_available = True
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from a JSON file (optional fallback)."""
        if not self.config_path:
            return {}
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config from {self.config_path}: {e}")
            return {}
        
    def _get_element_text(self, element: Dict[str, Any]) -> str:
        """Extract text content from an element."""
        content = element.get('content', '')
        element_type = element.get('type', '')
        
        if element_type == 'figure':
            description = element.get('description', '')
            if description:
                return f"[FIGURE] {content}\n{description}" if content else f"[FIGURE] {description}"
        
        if element_type == 'table':
            return f"[TABLE] {content}"
            
        if element_type in ['title', 'section_header']:
            return f"[{element_type.upper()}] {content}"
            
        return content
    
    def _is_section_break(self, element: Dict[str, Any]) -> bool:
        """Check if an element represents a section break."""
        element_type = element.get('type', '')
        return element_type in ['section_header', 'title']
    
    def _get_tokenizer(self):
        """Get or initialize the tokenizer for the embedding model."""
        if self._tokenizer is None and self._tokenizer_available:
            try:
                # Try to get tokenizer from embedding_config first
                tokenizer_factory = get_embedding_model_tokenizer(self.embedding_model, self.cache_dir)
                
                if tokenizer_factory:
                    self._tokenizer = tokenizer_factory()
                    logging.info(f"Initialized tokenizer for {self.embedding_model} using embedding_config")
                else:
                    # Fallback to JSON config if available
                    embedding_models = self.config.get('embedding_models_with_tokenizer', {})
                    model_config = embedding_models.get(self.embedding_model)
                    
                    if model_config and 'model' in model_config:
                        from transformers import AutoTokenizer
                        model_name = model_config['model']
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            cache_dir=self.cache_dir
                        )
                        logging.info(f"Initialized tokenizer for {self.embedding_model} using model {model_name} from config")
                    else:
                        # Fallback to character-based estimation
                        self._tokenizer_available = False
                        logging.warning(f"Tokenizer not available for {self.embedding_model}, falling back to character-based estimation")
                        
            except Exception as e:
                self._tokenizer_available = False
                logging.warning(f"Failed to initialize tokenizer: {e}, falling back to character-based estimation")
        
        return self._tokenizer
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count using the embedding model's tokenizer or fallback to character-based estimation."""
        if not text or not text.strip():
            return 0
            
        # Try to use the proper tokenizer first
        if self._tokenizer_available:
            try:
                tokenizer = self._get_tokenizer()
                if tokenizer:
                    # Use the tokenizer to get accurate token count
                    # Handle both HuggingFace tokenizers and tiktoken encodings
                    if hasattr(tokenizer, 'encode'):
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                    else:
                        # tiktoken encoding object
                        tokens = tokenizer.encode(text)
                    return len(tokens)
            except Exception as e:
                logging.warning(f"Tokenizer failed: {e}, falling back to character-based estimation")
                self._tokenizer_available = False
        
        # Fallback to character-based estimation
        # More accurate ratios based on typical English text
        char_count = len(text)
        return char_count // 4
    
    def chunk_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main chunking function that processes elements and creates semantic chunks.
        
        Args:
            elements: List of element dictionaries from PDF parsing
            
        Returns:
            List of chunk dictionaries with basic metadata
        """
        if not elements:
            return []
        
        chunks = []
        current_chunk = []
        current_section = "Introduction"  # Default section
        current_size = 0
        
        for element in elements:
            element_type = element.get('type', '')
            element_text = self._get_element_text(element)
            element_size = self._estimate_token_count(element_text)
            
            # Check if this is a section break
            if self._is_section_break(element):
                # Save current chunk if it has content and meets minimum size
                if current_chunk and current_size >= self.min_chunk_size:
                    chunks.append({
                        'elements': current_chunk,
                        'section': current_section,
                        'chunk_type': 'section_based'
                    })
                elif current_chunk:
                    # If chunk is too small, try to merge with previous chunk
                    if chunks and chunks[-1]['chunk_type'] in ['section_based', 'size_limited', 'final', 'small_section', 'merged']:
                        # Merge with previous chunk
                        chunks[-1]['elements'].extend(current_chunk)
                        chunks[-1]['chunk_type'] = 'merged'
                    else:
                        # If no previous chunk to merge with, save as is
                        chunks.append({
                            'elements': current_chunk,
                            'section': current_section,
                            'chunk_type': 'small_section'
                        })
                
                # Start new chunk with the section header
                current_section = element_text.replace(f"[{element_type.upper()}] ", "")
                current_chunk = [element]
                current_size = element_size
                
            else:
                # Check if adding this element would exceed size limit
                if (current_size + element_size > self.max_chunk_size and 
                    current_chunk and 
                    element_type not in self.atomic_types):
                    
                    # Save current chunk if it meets minimum size
                    if current_size >= self.min_chunk_size:
                        chunks.append({
                            'elements': current_chunk,
                            'section': current_section,
                            'chunk_type': 'size_limited'
                        })
                    else:
                        # Try to merge with previous chunk if current is too small
                        if chunks and chunks[-1]['chunk_type'] in ['section_based', 'size_limited', 'final', 'small_section', 'merged']:
                            chunks[-1]['elements'].extend(current_chunk)
                            chunks[-1]['chunk_type'] = 'merged'
                        else:
                            chunks.append({
                                'elements': current_chunk,
                                'section': current_section,
                                'chunk_type': 'small_size'
                            })
                    
                    # Start new chunk
                    current_chunk = [element]
                    current_size = element_size
                    
                else:
                    # Add element to current chunk
                    current_chunk.append(element)
                    current_size += element_size
        
        # Add final chunk - check if it meets minimum size
        if current_chunk:
            if current_size >= self.min_chunk_size:
                chunks.append({
                    'elements': current_chunk,
                    'section': current_section,
                    'chunk_type': 'final'
                })
            else:
                # Try to merge with previous chunk if final chunk is too small
                if chunks and chunks[-1]['chunk_type'] in ['section_based', 'size_limited', 'final', 'small_section', 'merged']:
                    chunks[-1]['elements'].extend(current_chunk)
                    chunks[-1]['chunk_type'] = 'merged'
                else:
                    chunks.append({
                        'elements': current_chunk,
                        'section': current_section,
                        'chunk_type': 'small_final'
                    })
        
        return chunks
    
    def _get_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Get the full text content of a chunk."""
        return "\n\n".join([
            self._get_element_text(element) 
            for element in chunk['elements']
        ])


def get_chunk_schema() -> ArrayType:
    """Get the Spark schema for chunked elements."""
    return ArrayType(StructType([
        StructField('content', StringType(), True),
        StructField('section', StringType(), True),
        StructField('chunk_type', StringType(), True)
    ]))


def get_elements_chunking_udf(options: ElementsChunkerOptions):
    """
    Create a Spark UDF for element-based chunking.
    
    Args:
        options: ElementsChunkerOptions configuration
        
    Returns:
        Spark UDF function
    """
    def chunking_udf(element_json_str: str) -> List[Dict[str, Any]]:
        """UDF function that processes JSON element string and returns chunks."""
        if not element_json_str:
            return []
        try:
            elements_data = json.loads(element_json_str)

            if isinstance(elements_data, dict) and 'elements' in elements_data:
                elements_list = elements_data['elements']
            elif isinstance(elements_data, list):
                elements_list = elements_data
            else:
                elements_list = [elements_data]
        
            chunker = ElementsChunker(options)
            chunks = chunker.chunk_elements(elements_list)
            
            # Create output
            result = []
            for i, chunk in enumerate(chunks):
                chunk_text = chunker._get_chunk_text(chunk)
                result.append({
                    'content': chunk_text,
                    'section': chunk['section'],
                    'chunk_type': chunk['chunk_type']
                })
                    
            return result

        except Exception as e:
            logging.error(f"Error parsing JSON in chunking UDF: {e}")
            return []
    
    chunk_schema = get_chunk_schema()
    return udf(chunking_udf, chunk_schema)

