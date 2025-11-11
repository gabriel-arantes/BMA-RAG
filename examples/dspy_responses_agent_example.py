# Databricks notebook source
# MAGIC %pip install --index-url https://pypi.org/simple dspy databricks-agents mlflow[databricks]>=3.1.0 databricks-vectorsearch databricks-sdk databricks-mcp databricks-dspy uv
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # DSPy ResponsesAgent Example
# MAGIC 
# MAGIC This notebook demonstrates how to create a **DSPy-based ResponsesAgent** that:
# MAGIC - Uses the ReAct pattern with vector search and fallback tools
# MAGIC - Supports multi-turn conversation history
# MAGIC - Integrates with MLflow for tracing and logging
# MAGIC - Follows Databricks Agent Framework patterns
# MAGIC 
# MAGIC ## When to Use ResponsesAgent vs RAG
# MAGIC 
# MAGIC **Use ResponsesAgent when:**
# MAGIC - You need interactive chat with tool-calling capabilities
# MAGIC - You want the agent to decide which tools to use (ReAct pattern)
# MAGIC - You need multi-turn conversation support with history management
# MAGIC - You're building a chatbot or conversational interface
# MAGIC 
# MAGIC **Use RAG pattern when:**
# MAGIC - You need a simpler retrieval-augmented generation pipeline
# MAGIC - You want to optimize prompts and demonstrations (GEPA, MIPROv2)
# MAGIC - You need to compare different metrics during training
# MAGIC - You're building a Q&A system without complex tool-calling needs

# COMMAND ----------

import dspy
import mlflow
import os
import uuid
from typing import Optional
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker
from mlflow.models import set_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC 
# MAGIC Set up widgets for configuration parameters

# COMMAND ----------

# Configuration widgets
dbutils.widgets.text("chat_endpoint_name", "databricks-claude-3-7-sonnet", "Chat Endpoint Name")
dbutils.widgets.text("vector_search_endpoint", "ctcbl-unstructured-endpoint", "Vector Search Endpoint")
dbutils.widgets.text("catalog_name", "test_catalog", "Catalog Name")
dbutils.widgets.text("schema_name", "test_schema", "Schema Name")
dbutils.widgets.text("vector_index_name", "test_volume_ctcbl_chunked_index_element__v0_0_1", "Vector Index Name")
dbutils.widgets.text("max_history_length", "10", "Max History Length")
dbutils.widgets.text("enable_history", "true", "Enable History")

# Get widget values
model = dbutils.widgets.get("chat_endpoint_name")
LM = f"databricks/{model}"
VECTOR_SEARCH_ENDPOINT = dbutils.widgets.get("vector_search_endpoint")
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
VECTOR_SEARCH_INDEX = dbutils.widgets.get("vector_index_name")
INDEX_PATH = f"{CATALOG}.{SCHEMA}.{VECTOR_SEARCH_INDEX}"
max_history_length = int(dbutils.widgets.get("max_history_length"))
enable_history = dbutils.widgets.get("enable_history").lower() == "true"

print(f"✅ Configuration loaded:")
print(f"   LLM: {LM}")
print(f"   Vector Index: {INDEX_PATH}")
print(f"   Max History: {max_history_length}")
print(f"   History Enabled: {enable_history}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Core Utilities
# MAGIC 
# MAGIC ### History Extraction
# MAGIC 
# MAGIC The `extract_history_from_messages` function converts a list of messages into DSPy History format.
# MAGIC It pairs user messages with assistant responses and maintains conversation context.

# COMMAND ----------

def extract_history_from_messages(messages, max_length=None, enable_history=True):
    """
    Extract conversation history from message list.
    Converts [{role, content}, ...] to dspy.History format.
    
    Args:
        messages: List of message dicts/objects with role and content
        max_length: Maximum number of conversation turns to include
        enable_history: Whether to extract history (if False, returns empty history)
    
    Returns:
        dspy.History object with messages in format [{"question": ..., "response": ...}]
    """
    if not enable_history or not messages or len(messages) <= 1:
        return dspy.History(messages=[])
    
    history_messages = []
    
    # Extract previous messages (excluding last one, which is the current question)
    previous_messages = messages[:-1]
    
    # Pair user messages with following assistant messages
    i = 0
    while i < len(previous_messages):
        # Get message content, handling both dict and object formats
        msg = previous_messages[i]
        role = msg.role if hasattr(msg, "role") else msg.get("role", "")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        
        if role == "user" and content:
            # Look for following assistant message
            if i + 1 < len(previous_messages):
                next_msg = previous_messages[i + 1]
                next_role = next_msg.role if hasattr(next_msg, "role") else next_msg.get("role", "")
                next_content = next_msg.content if hasattr(next_msg, "content") else next_msg.get("content", "")
                
                if next_role == "assistant" and next_content:
                    # Found a user/assistant pair
                    history_messages.append({
                        "question": content,
                        "response": next_content
                    })
                    i += 2  # Skip both messages
                    continue
        
        i += 1
    
    # Apply max_length if specified (keep most recent turns)
    if max_length is not None and max_length > 0:
        history_messages = history_messages[-max_length:]
    
    # Create history object
    history = dspy.History(messages=history_messages)
    
    # Optional debug output (controlled by DSPY_DEBUG_HISTORY environment variable)
    if os.environ.get("DSPY_DEBUG_HISTORY", "false").lower() == "true":
        print(f"[DEBUG] Extracted history: {len(history_messages)} turns")
        print(f"[DEBUG] History messages: {history_messages}")
    
    return history

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Search Tool
# MAGIC 
# MAGIC The `VectorSearchTool` wraps Databricks Vector Search as a DSPy tool,
# MAGIC allowing the agent to search for relevant documents dynamically.

# COMMAND ----------

class VectorSearchTool:
    """Custom Vector Search tool wrapper for DSPy"""

    def __init__(
        self,
        endpoint_name: str,
        index_name: str,
        tool_name: str = "vector_search",
        description: str = "Search for relevant documents using vector similarity",
        num_results: int = 3
    ):
        self.endpoint_name = endpoint_name
        self.index_name = index_name
        self.tool_name = tool_name
        self.description = description
        self.num_results = num_results

        # Initialize Vector Search client
        self.vs_client = VectorSearchClient()
        self.index = self.vs_client.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )

    def search(self, query: str) -> str:
        """
        Perform vector search and return formatted results as a string.
        
        Args:
            query: The search query string
            
        Returns:
            str: A formatted string representation of search results, or an error message
        """
        if not query or not query.strip():
            return "Error: Empty query provided"
            
        try:
            results = self.index.similarity_search(
                query_text=query,
                columns=["chunk_id", "chunk_content", "path"],
                num_results=self.num_results,
                reranker=DatabricksReranker(
                    columns_to_rerank=["chunk_content"]
                )
            )

            hits = results['result']['data_array']
            if not hits:
                return "No results found for the query."

            columns = [col['name'] for col in results['manifest']['columns']]
            result_dicts = [dict(zip(columns, row)) for row in hits]
            
            # Format results as a readable string for DSPy
            formatted_results = []
            for idx, result in enumerate(result_dicts, 1):
                formatted_results.append(
                    f"Result {idx}:\n"
                    f"  Content: {result.get('chunk_content', 'N/A')}\n"
                    f"  Source: {result.get('path', 'N/A')}\n"
                    f"  Chunk ID: {result.get('chunk_id', 'N/A')}"
                )
            return "\n\n".join(formatted_results)

        except Exception as e:
            return f"Error performing vector search: {str(e)}"

# Create Vector Search tool instance
vector_search_tool = VectorSearchTool(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_PATH,
    tool_name="vector_search",
    description="Search for relevant documents using vector similarity",
    num_results=3
)

# Convert to DSPy tool
dspy_vector_search_tool = dspy.Tool(
    func=vector_search_tool.search,
    name="vector_search",
    desc="Search for relevant documents using vector similarity"
)

print("✅ Vector search tool initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## DSPy Signature
# MAGIC 
# MAGIC The `BMAChatAssistant` signature defines the input/output structure for the agent.
# MAGIC It includes context, question, and history fields.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## DSPyChatAgent Implementation
# MAGIC 
# MAGIC The `DSPyChatAgent` class implements the ResponsesAgent interface using DSPy's ReAct pattern.
# MAGIC 
# MAGIC **Key Features:**
# MAGIC - Tool-calling with ReAct (Reasoning + Acting)
# MAGIC - Conversation history extraction and management
# MAGIC - Automatic MLflow tracing
# MAGIC - Fallback tool for handling unanswerable questions

# COMMAND ----------

class DSPyChatAgent(ResponsesAgent):
    """
    A DSPy-based responses agent that uses ReAct pattern with vector search and fallback tools.
    """
    
    @staticmethod
    def not_enough_info() -> str:
        """
        Tool called when the assistant is unable to answer the question.
        
        Returns:
            str: A message indicating that the assistant is unable to answer the question.
        """
        return ("I'm sorry, I don't have enough information to answer your question. "
                "Please ask the user to provide more details or ask a different question.")
    
    def __init__(self, vector_search_tool: Optional[dspy.Tool] = None, max_history_length=None, enable_history=True):
        """
        Initialize the DSPy responses agent.
        
        Args:
            vector_search_tool: Optional pre-configured vector search tool.
                                If None, uses the default dspy_vector_search_tool.
            max_history_length: Maximum number of conversation turns to include in history.
            enable_history: Whether to enable conversation history.
        """
        dspy.configure(lm=dspy.LM(LM))
        mlflow.dspy.autolog()
        
        self.BMAChatAssistant = BMAChatAssistant
        self.dspy_vector_search_tool = vector_search_tool or dspy_vector_search_tool
        # Use widget values as defaults if not provided
        self.max_history_length = max_history_length if max_history_length is not None else globals().get('max_history_length', 10)
        self.enable_history = enable_history if enable_history is not None else globals().get('enable_history', True)
        
        # Create not_enough_info tool using static method to avoid circular reference
        self.not_enough_info_tool = dspy.Tool(
            func=self.not_enough_info,
            name="not_enough_info",
            desc="This tool is called when the assistant is unable to answer the question."
        )
        
        self.answer_generator = dspy.ReAct(
            self.BMAChatAssistant,
            tools=[self.dspy_vector_search_tool, self.not_enough_info_tool]
        )
        # With the context field present in the signature, ReAct will no longer warn about missing 'context'.
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Generate a response to the messages.
        
        Args:
            request: ResponsesAgentRequest containing input messages
            
        Returns:
            ResponsesAgentResponse containing the assistant's response
            
        Raises:
            ValueError: If messages list is empty
        """
        # Convert the request input (list of messages) into your message list
        user_msgs = request.input  # list of Messages [{role,content},...]
        
        if not user_msgs:
            msg_id = uuid.uuid4().hex
            return ResponsesAgentResponse(
                output=[self.create_text_output_item(
                    text="Messages list cannot be empty",
                    id=msg_id
                )]
            )
        
        latest = user_msgs[-1]
        content = latest.content if hasattr(latest, "content") else latest.get("content", "")
        
        if not content or not content.strip():
            msg_id = uuid.uuid4().hex
            return ResponsesAgentResponse(
                output=[self.create_text_output_item(
                    text="I received an empty message. Could you please rephrase your question?",
                    id=msg_id
                )]
            )
        
        # Extract conversation history from messages
        history = extract_history_from_messages(user_msgs, self.max_history_length, self.enable_history)
        
        # Optional debug output (controlled by DSPY_DEBUG_HISTORY environment variable)
        if os.environ.get("DSPY_DEBUG_HISTORY", "false").lower() == "true":
            print(f"[DEBUG DSPyChatAgent] History: {len(history.messages)} turns, Question: {content}")
        
        # Use your DSPy generator
        # For this chat agent, we don't pre-supply retrieved context, so we pass an empty string.
        response_text = self.answer_generator(context="", question=content, history=history).response
        
        msg_id = uuid.uuid4().hex
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(
                text=response_text,
                id=msg_id
            )]
        )

print("✅ DSPyChatAgent class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 1: Single-Turn Conversation (No History)
# MAGIC 
# MAGIC Let's start with a simple single-turn question-answer interaction.

# COMMAND ----------

# Initialize the agent
agent = DSPyChatAgent(max_history_length=max_history_length, enable_history=enable_history)
set_model(agent)

# Single-turn question
single_turn_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What was the overall audit rating for CTCBL?"}]
)

response = agent.predict(single_turn_request)
print("🔹 Single-Turn Example")
print(f"Question: What was the overall audit rating for CTCBL?")
print(f"Response: {response.output[0].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 2: Multi-Turn Conversation (With History)
# MAGIC 
# MAGIC Now let's demonstrate how the agent maintains conversation context across multiple turns.

# COMMAND ----------

# Enable debug mode to see history extraction
os.environ["DSPY_DEBUG_HISTORY"] = "true"

# Multi-turn conversation
print("🔹 Multi-Turn Conversation Example")
print("(Debug mode enabled - you'll see history extraction)\n")

# Turn 1: Initial question
turn1_request = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "Who is responsible for compliance policy amendments?"}
    ]
)
turn1_response = agent.predict(turn1_request)
print("Turn 1:")
print(f"User: Who is responsible for compliance policy amendments?")
print(f"Assistant: {turn1_response.output[0].content}\n")

# Turn 2: Follow-up question (uses history)
turn2_request = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "Who is responsible for compliance policy amendments?"},
        {"role": "assistant", "content": turn1_response.output[0].content},
        {"role": "user", "content": "What is their role exactly?"}
    ]
)
turn2_response = agent.predict(turn2_request)
print("Turn 2:")
print(f"User: What is their role exactly?")
print(f"Assistant: {turn2_response.output[0].content}\n")

# Turn 3: Another follow-up
turn3_request = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "Who is responsible for compliance policy amendments?"},
        {"role": "assistant", "content": turn1_response.output[0].content},
        {"role": "user", "content": "What is their role exactly?"},
        {"role": "assistant", "content": turn2_response.output[0].content},
        {"role": "user", "content": "What other compliance-related responsibilities exist?"}
    ]
)
turn3_response = agent.predict(turn3_request)
print("Turn 3:")
print(f"User: What other compliance-related responsibilities exist?")
print(f"Assistant: {turn3_response.output[0].content}")

# Disable debug mode
os.environ["DSPY_DEBUG_HISTORY"] = "false"
print("\n✅ Multi-turn conversation completed - history was extracted for turns 2 and 3")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 3: Advanced Multi-Turn with Debug Mode
# MAGIC 
# MAGIC Enable debug mode to see how history is being extracted and processed with a longer conversation.

# COMMAND ----------

# Enable debug mode
os.environ["DSPY_DEBUG_HISTORY"] = "true"

print("🔹 Advanced Debug Mode Example")
print("Longer conversation with 3 Q&A pairs in history\n")

# Create a conversation with multiple turns of history
advanced_debug_request = ResponsesAgentRequest(
    input=[
        # Turn 1
        {"role": "user", "content": "What was the audit period?"},
        {"role": "assistant", "content": "The audit covered April 1, 2021 to October 31, 2022."},
        # Turn 2
        {"role": "user", "content": "What was the overall audit rating?"},
        {"role": "assistant", "content": "The overall audit rating for CTCBL was Satisfactory."},
        # Turn 3
        {"role": "user", "content": "Were there any major findings?"},
        {"role": "assistant", "content": "Yes, there was one major finding: Noncompliance with Policies and Procedures, which was rated as Medium."},
        # Turn 4 - Current question
        {"role": "user", "content": "Who is responsible for remediation?"}
    ]
)

print("📊 This request has 3 Q&A pairs in history (turns 1-3) plus current question (turn 4)")
print("You should see '[DEBUG] Extracted history: 3 turns' in the output below:\n")

advanced_debug_response = agent.predict(advanced_debug_request)
print(f"\n✅ Final Response: {advanced_debug_response.output[0].content}")

# Disable debug mode
os.environ["DSPY_DEBUG_HISTORY"] = "false"
print("\n🔍 Debug output above shows history extraction working correctly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 4: Custom Vector Search Configuration
# MAGIC 
# MAGIC You can customize the vector search tool with different parameters.

# COMMAND ----------

# Create a custom vector search tool with more results
custom_vector_tool = VectorSearchTool(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_PATH,
    tool_name="detailed_search",
    description="Search for relevant documents with more results",
    num_results=5  # Return more results
)

custom_dspy_tool = dspy.Tool(
    func=custom_vector_tool.search,
    name="detailed_search",
    desc="Search for relevant documents with more detailed results"
)

# Create agent with custom tool
custom_agent = DSPyChatAgent(
    vector_search_tool=custom_dspy_tool,
    max_history_length=5,  # Shorter history
    enable_history=True
)

print("✅ Custom agent created with 5 search results and max 5 history turns")

custom_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What were the key audit findings?"}]
)
custom_response = custom_agent.predict(custom_request)
print(f"\nResponse: {custom_response.output[0].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 5: Error Handling
# MAGIC 
# MAGIC The agent gracefully handles edge cases like empty messages.

# COMMAND ----------

print("🔹 Error Handling Examples\n")

# Empty message list
try:
    empty_request = ResponsesAgentRequest(input=[])
    empty_response = agent.predict(empty_request)
    print(f"Empty message list: {empty_response.output[0].content}\n")
except Exception as e:
    print(f"Error with empty list: {e}\n")

# Empty content
empty_content_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": ""}]
)
empty_content_response = agent.predict(empty_content_request)
print(f"Empty content: {empty_content_response.output[0].content}\n")

# Whitespace only
whitespace_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "   "}]
)
whitespace_response = agent.predict(whitespace_request)
print(f"Whitespace only: {whitespace_response.output[0].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 6: MLflow Integration
# MAGIC 
# MAGIC The agent automatically logs traces to MLflow via `mlflow.dspy.autolog()`.

# COMMAND ----------

import mlflow

print("🔹 MLflow Integration Example\n")

# Set experiment (optional, for better organization)
mlflow.set_experiment("/Workspace/Users/garantes_oc@bma.bm/dspy_responses_agent_example")

# Make a prediction - it will be automatically traced
with mlflow.start_run(run_name="example_prediction"):
    mlflow_request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Who is responsible for trigger event policy updates?"}]
    )
    mlflow_response = agent.predict(mlflow_request)
    
    # Log additional metadata
    mlflow.log_param("max_history_length", max_history_length)
    mlflow.log_param("enable_history", enable_history)
    mlflow.log_param("model", model)
    
    print(f"Question: Who is responsible for trigger event policy updates?")
    print(f"Response: {mlflow_response.output[0].content}")
    print(f"\n✅ Prediction traced to MLflow!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook demonstrated:
# MAGIC 
# MAGIC 1. ✅ **Setup**: Installation, imports, and configuration
# MAGIC 2. ✅ **Core Components**: History extraction, vector search tool, signature
# MAGIC 3. ✅ **DSPyChatAgent**: ResponsesAgent implementation with ReAct pattern
# MAGIC 4. ✅ **Single-Turn**: Simple Q&A without history
# MAGIC 5. ✅ **Multi-Turn**: Conversation with context awareness (with debug output)
# MAGIC 6. ✅ **Advanced Debug Mode**: Longer conversation demonstrating history extraction
# MAGIC 7. ✅ **Custom Configuration**: Adjusting search and history parameters
# MAGIC 8. ✅ **Error Handling**: Graceful handling of edge cases
# MAGIC 9. ✅ **MLflow Integration**: Automatic tracing and logging
# MAGIC 
# MAGIC ### Key Takeaways
# MAGIC 
# MAGIC - **ResponsesAgent** is perfect for interactive chat with tool-calling
# MAGIC - **History management** enables context-aware conversations
# MAGIC - **ReAct pattern** allows the agent to reason about which tools to use
# MAGIC - **MLflow integration** provides observability and tracing
# MAGIC 
# MAGIC ### Next Steps
# MAGIC 
# MAGIC - Deploy the agent as a model serving endpoint
# MAGIC - Integrate with Agent Evaluation Suite
# MAGIC - Add custom tools for your specific use case
# MAGIC - Fine-tune prompts and instructions for better performance

