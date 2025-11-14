# dspy_program.py
import dspy
import os
import mlflow
import pandas as pd
from dspy.retrievers.databricks_rm import DatabricksRM
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker
from typing import Optional

def extract_history_from_messages(messages, max_length=None, enable_history=True):
    """
    Extract conversation history from message list.
    Converts [{role, content}, ...] to dspy.History format.
    """
    if not enable_history or not messages or len(messages) <= 1:
        return dspy.History(messages=[])
    
    history_messages = []
    previous_messages = messages[:-1]
    i = 0
    while i < len(previous_messages):
        msg = previous_messages[i]
        role = msg.role if hasattr(msg, "role") else msg.get("role", "")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        
        if role == "user" and content:
            if i + 1 < len(previous_messages):
                next_msg = previous_messages[i + 1]
                next_role = next_msg.role if hasattr(next_msg, "role") else next_msg.get("role", "")
                next_content = next_msg.content if hasattr(next_msg, "content") else next_msg.get("content", "")
                
                if next_role == "assistant" and next_content:
                    history_messages.append({
                        "question": content,
                        "response": next_content
                    })
                    i += 2
                    continue
        i += 1
    
    if max_length is not None and max_length > 0:
        history_messages = history_messages[-max_length:]
    
    return dspy.History(messages=history_messages)

class BMAChatAssistant(dspy.Signature):
    """
    You are a trusted assistant that helps answer questions based only on the provided information. 
    You are given a list of tools to handle the customer's request. You should decide the right tool 
    to use in order to appropriately answer the customer's question.
    """
    context: str = dspy.InputField(desc="The retrieved or provided context to answer the customer's question.")
    question: str = dspy.InputField(desc="The customer's question that needs to be answered.")
    history: dspy.History = dspy.InputField(desc="A record of previous conversation turns as question/response pairs.")
    response: str = dspy.OutputField(desc="The assistant's answer to the customer's question, based solely on the context.")

class RAG(dspy.Module):
    """
    Stateless, fully serializable DSPy RAG module.
    Does NOT capture non-serializable Databricks objects or globals.
    """
    
    def __init__(
        self,
        lm_name: str,
        index_path: str,
        max_history_length: int = 10,
        enable_history: bool = True,
        for_mosaic_agent: bool = True
    ):
        super().__init__()
        self.lm_name = lm_name
        self.index_path = index_path
        self.max_history_length = max_history_length
        self.enable_history = enable_history
        self.for_mosaic_agent = for_mosaic_agent
        self.response_generator = dspy.Predict(BMAChatAssistant)
    
    def build_retriever(self):
        return DatabricksRM(
            databricks_index_name=self.index_path,
            text_column_name="chunk_content",
            docs_id_column_name="chunk_id",
            columns=["chunk_id", "chunk_content", "path"],
            k=5,
            use_with_databricks_agent_framework=self.for_mosaic_agent,
        )
    
    def build_lm(self):
        return dspy.LM(model=f"databricks/{self.lm_name}")
    
    def forward(self, messages):
        if isinstance(messages, dict) and "messages" in messages:
            message_list = messages["messages"]
            question_text = message_list[-1]["content"]
            history = extract_history_from_messages(
                message_list,
                max_length=self.max_history_length,
                enable_history=self.enable_history
            )
        else:
            # Fallback corrigido para usar 'messages'
            question_text = str(messages)
            history = dspy.History(messages=[])
        
        retriever = self.build_retriever()
        lm = self.build_lm()
        context = retriever(question_text)
        
        with dspy.context(lm=lm):
            return self.response_generator(
                context=context,
                question=question_text,
                history=history
            )