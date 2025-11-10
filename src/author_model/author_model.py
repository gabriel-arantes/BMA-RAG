# Databricks notebook source
# MAGIC %pip install --index-url https://pypi.org/simple dspy databricks-agents mlflow[databricks]>=3.1.0 databricks-vectorsearch databricks-sdk databricks-mcp databricks-dspy uv
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import dspy
import mlflow
import random 
import os
import uuid
from dspy.retrievers.databricks_rm import DatabricksRM
from typing import Any, Optional
from databricks.sdk.service.dashboards import GenieAPI # not used?
from databricks.sdk import WorkspaceClient # not used?
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker
from mlflow.models import set_model

# COMMAND ----------

# MAGIC %md
# MAGIC # AUTHOR AGENT USING DSPY

# COMMAND ----------

# Get parameters from widgets
dbutils.widgets.text("experiment_name", "/Workspace/Users/garantes_oc@bma.bm/bma_author_dspy_model_experiment", "Experiment Name")
dbutils.widgets.text("catalog_name", "test_catalog", "Catalog Name")
dbutils.widgets.text("schema_name", "test_schema", "Schema Name")
dbutils.widgets.text("vector_search_endpoint", "ctcbl-unstructured-endpoint", "Vector Search Endpoint")
dbutils.widgets.text("vector_index_name", "test_volume_ctcbl_chunked_index_element__v0_0_1", "Vector Index Name")
dbutils.widgets.text("evaluation_dataset_table", "rag_eval_dataset", "Evaluation Dataset Table")
dbutils.widgets.text("chat_endpoint_name", "databricks-claude-3-7-sonnet", "Chat Endpoint Name")
dbutils.widgets.text("small_chat_endpoint_name", "databricks-gpt-oss-20b", "Small Chat Endpoint Name")
dbutils.widgets.text("larger_chat_endpoint_name", "databricks-claude-3-7-sonnet", "Larger Chat Endpoint Name")
dbutils.widgets.text("reflection_chat_endpoint_name", "databricks-claude-sonnet-4-5", "Reflection Chat Endpoint Name")
dbutils.widgets.text("max_history_length", "10", "Max History Length")
dbutils.widgets.text("enable_history", "true", "Enable History")

# Get widget values
experiment_name = dbutils.widgets.get("experiment_name")
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
VECTOR_SEARCH_ENDPOINT = dbutils.widgets.get("vector_search_endpoint")
VECTOR_SEARCH_INDEX = dbutils.widgets.get("vector_index_name")
INDEX_PATH = f"{CATALOG}.{SCHEMA}.{VECTOR_SEARCH_INDEX}"
evaluation_dataset_tbl_name = dbutils.widgets.get("evaluation_dataset_table")

model = dbutils.widgets.get("chat_endpoint_name")
LM = f"databricks/{model}"
small_lm_name = dbutils.widgets.get("small_chat_endpoint_name")
larger_lm_name = dbutils.widgets.get("larger_chat_endpoint_name")
reflection_lm_name = dbutils.widgets.get("reflection_chat_endpoint_name")
max_history_length = int(dbutils.widgets.get("max_history_length"))
enable_history = dbutils.widgets.get("enable_history").lower() == "true"

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Experiment Setup

# COMMAND ----------

# Set up MLflow experiment (best practice)
mlflow.set_experiment(experiment_name)

# COMMAND ----------

vsc = VectorSearchClient(disable_notice=True)
question = "Who is responsible for remediating the issues with the Customer Risk Assessment for the high-risk PEP client, and what is the remediation date?"
try:
    results = vsc.get_index( 
                endpoint_name=VECTOR_SEARCH_ENDPOINT,
                index_name=INDEX_PATH
            ).similarity_search(
                query_text=question,
                columns=["chunk_id", "chunk_content", "path"],
                num_results=5,
                reranker=DatabricksReranker(
                    columns_to_rerank=["chunk_content"]
                )
            )
    print(results)
    data_array = results['result']['data_array']
    columns = [col['name'] for col in results['manifest']['columns']]
    print([dict(zip(columns, row)) for row in data_array])

except Exception as e:
    print(f"Error querying vector index: {e}")
    

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

# Create Vector Search tool
vector_search_tool = VectorSearchTool(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_PATH,
    tool_name="vector_search",
    description="Search for relevant documents using vector similarity",
    num_results=3
)

# Convert Vector Search tool to DSPy tool
dspy_vector_search_tool = dspy.Tool(
    func=vector_search_tool.search,
    name="vector_search",
    desc="Search for relevant documents using vector similarity"
)

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

# Set model for logging or interactive testing
AGENT = DSPyChatAgent(max_history_length=max_history_length, enable_history=enable_history)
set_model(AGENT)

# Test the agent with ResponsesAgent API
# ResponsesAgentRequest expects input as a list of message dicts or objects
test_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Who is responsible for remediating the issues with the Customer Risk Assessment for the super low-risk PEP client, and what is the remediation date?"}]
)
AGENT.predict(test_request)

# COMMAND ----------

class RAG(dspy.Module):
    def __init__(self, lm_name, for_mosaic_agent=True, max_history_length=None, enable_history=True):
        # setup mlflow tracing
        mlflow.dspy.autolog()

        # setup flag indicating if the object will be deployed as a Mosaic Agent
        self.for_mosaic_agent = for_mosaic_agent
        self.lm = dspy.LM(model=f"databricks/{lm_name}")
        # Use widget values as defaults if not provided
        self.max_history_length = max_history_length if max_history_length is not None else globals().get('max_history_length', 10)
        self.enable_history = enable_history if enable_history is not None else globals().get('enable_history', True)

        # setup the primary retriever pointing to the chunked documents
        self.retriever = DatabricksRM(
            databricks_index_name=INDEX_PATH,
            text_column_name="chunk_content",
            docs_id_column_name="chunk_id",
            columns=["chunk_id", "chunk_content", "path"],
            k=5,
            use_with_databricks_agent_framework=for_mosaic_agent,
        )

        # Reuse the same BMAChatAssistant signature with context
        self.response_generator = dspy.Predict(BMAChatAssistant)

    def forward(self, question):
        # Preserve full question structure for history extraction
        original_question = question
        
        # Extract question text for retrieval
        if self.for_mosaic_agent:
            if isinstance(question, dict) and "messages" in question:
                question_text = question["messages"][-1]["content"]
                # Extract history from messages
                history = extract_history_from_messages(question["messages"], self.max_history_length, self.enable_history)
            else:
                question_text = question
                history = dspy.History(messages=[])
        else:
            # For non-mosaic agent, extract history if question is a dict with messages
            if isinstance(question, dict) and "messages" in question:
                question_text = question["messages"][-1]["content"] if question["messages"] else str(question)
                history = extract_history_from_messages(question["messages"], self.max_history_length, self.enable_history)
            else:
                question_text = question
                history = dspy.History(messages=[])

        # Optional debug output (controlled by DSPY_DEBUG_HISTORY environment variable)
        if os.environ.get("DSPY_DEBUG_HISTORY", "false").lower() == "true":
            print(f"[DEBUG RAG] History: {len(history.messages)} turns, Question: {question_text}")

        # Update trace tags if there's an active trace context
        # Note: mlflow.dspy.autolog() creates traces automatically during normal DSPy operations,
        # but during optimization bootstrapping (MIPROv2/GEPA), forward() may be called
        # without an active trace, causing warnings. Check for active trace before updating.
        try:
            # Check if there's an active trace before trying to update
            trace_id = mlflow.tracing.get_current_trace_id()
            if trace_id is not None:
                mlflow.update_current_trace(tags={"agent": "dspy_rag_demo"})
        except (AttributeError, Exception):
            # MLflow tracing API might not be available or no active trace
            pass
        
        context = self.retriever(question_text)

        with dspy.context(lm=self.lm):
            response = self.response_generator(context=context, question=question_text, history=history)
            if self.for_mosaic_agent:
                return response
            return response

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an instance of the agent and vibe check

# COMMAND ----------


rag = RAG(lm_name=small_lm_name, max_history_length=max_history_length, enable_history=enable_history)
simple_rag = RAG(lm_name=small_lm_name, for_mosaic_agent=False, max_history_length=max_history_length, enable_history=enable_history)

result = rag({'messages': [{'content': 'Who is responsible for remediating the issues with the Customer Risk Assessment for the high-risk PEP client, and what is the remediation date?', 'role': 'user'}]})

# COMMAND ----------

# MAGIC %md
# MAGIC # OPTIMIZATION
# MAGIC ## Manipulating examples in DSPy
# MAGIC To measure the quality of your DSPy system, you need (1) a bunch of input values, like questions for example, and (2) a metric that can score the quality of an output from your system. Metrics vary widely. Some metrics need ground-truth labels of ideal outputs, e.g. for classification or question answering. Other metrics are self-supervised, e.g. checking faithfulness or lack of hallucination, perhaps using a DSPy program as a judge of these qualities.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generating some fake ground truth labels or ideal outputs

# COMMAND ----------

# Raw training data
raw_training_data = [
  {"question":"Which period did the AML/ATF audit cover?","response":"The audit covered April 1, 2021 to October 31, 2022."},
  {"question":"What was the overall audit rating for CTCBL?","response":"Satisfactory."},
  {"question":"List the seven main pillars of the AML/ATF Program.","response":"Risk-Based Approach, Customer Due Diligence, Ongoing monitoring, Suspicious Activity Reporting, Record-Keeping, Training, Outsourcing and Reliance."},
  {"question":"Name one major audit finding and its rating.","response":"\"Noncompliance with Policies and Procedures\" was rated as Medium."},
  {"question":"Who is responsible for compliance policy amendments?","response":"Elkeisha Caisey, Senior Compliance Specialist."},
  {"question":"What is one area where internal controls need improvement?","response":"Internal controls and processes need to be performed in a consistent and timely manner."},
  {"question":"What were the main findings for Entity 35251?","response":"The Trigger Event Form and CRA were not completed or on file; ViewPoint was not updated for the retiring Protector."},
  {"question":"Who remediated the findings for Entity 35251?","response":"Natasha Hernandez, Head of Trust Services, CTCBL."},
  {"question":"What issue was found with distribution documentation for Entity 41683?","response":"Signed distribution task lists and Memorandum for the 6th distribution were missing; one distribution amount was incorrectly stated."},
  {"question":"How must distribution requests be verified?","response":"Signature verification via callback, Distribution Authentication stamp, Exception to Policy Form if needed, and minimum 3 signatures with one from Group Compliance."},
  {"question":"Who is responsible for distribution issue remediation?","response":"Natasha Hernandez, Head of Trust Services, CTCBL."},
  {"question":"What is the policy for employee screenings under AML/CTF?","response":"Criminal background checks, education and employment verification, and sanctions screening."},
  {"question":"What system supports AML/ATF training monitoring?","response":"A Learning and Management System (LMS) with robust tracking/monitoring capability."},
  {"question":"Who oversees the Learning and Management System implementation?","response":"Shea-Tai Smith, Senior Manager, Regulatory Risk."},
  {"question":"What improvements were recommended for the Global Compliance Framework?","response":"Update to clarify Board approval needs, provide updates for policy changes, include policies in annual Knowledge Management review."},
  {"question":"Who is the responsible person for Global Compliance Framework updates?","response":"Lanan Bascombe, Global Head of Compliance."},
  {"question":"What is the remediation deadline for the Global Compliance Framework?","response":"July 31, 2023."},
  {"question":"When are compliance policies approved?","response":"Annually by the Governance, Risk Compliance Committee or the Board."},
  {"question":"What is the main objective of the AML/CTF Policy?","response":"To meet BMA regulations and mitigate money laundering/terrorist financing risks."},
  {"question":"What methodology is used in audit assessment?","response":"Inquiry, observation, examination/inspection, and re-performance."},
  {"question":"How is ongoing automatic client screening performed?","response":"Weekly screening for all clients and semi-weekly for high-risk clients; reviews by specialists within five business days."},
  {"question":"How does CTCBL perform its ELRA?","response":"Using an Entity-Level Risk Assessment Tool to assess ML/TF risks and link controls."},
  {"question":"What change will be made to the ELRA after the audit?","response":"Language clarifying how controls mitigate each risk's inherent levels will be added."},
  {"question":"Who is responsible for ELRA updates?","response":"Shea-Tai Smith, Senior Manager, Regulatory Risk."},
  {"question":"What recommendation was made regarding trigger event documentation?","response":"Policy will be updated to specify a timeframe for when a trigger event may be part of a periodic review."},
  {"question":"Who is responsible for trigger event policy update?","response":"Ryan Yarde, Manager, Client Compliance Bermuda."},
  {"question":"List one significant gap identified between AML/CTF Policy and regulations.","response":"Omission of Regulation 10, paragraph 2(a) regarding simplified due diligence for regulated financial institutions."},
  {"question":"How should missing World-Check searches be remediated?","response":"Upload all screenings to client profiles."},
  {"question":"Which rating definitions apply to audit findings?","response":"High, Medium, Low - High means potential for material errors/losses; Medium for action to avoid regulatory or financial loss; Low for improvement opportunities."},
  {"question":"What training requirements exist for employees?","response":"Onboarding and annual AML/ATF and Sanctions training for all employees, including contractors."},
  {"question":"What issue was found among new employees in 2021?","response":"Missing employment and education verification for some new employees."},
  {"question":"What action should be taken for client CDD gaps?","response":"Remediate findings, provide refresher training, and ensure documentation is complete."},
  {"question":"Where are audit meeting minutes saved?","response":"GRCC and CTCBL Meeting Minutes are part of reviewed documentation."},
  {"question":"Which form is required for new client approval?","response":"New Business Approval Form."},
  {"question":"Which documentation relating to AML/ATF policies was reviewed?","response":"AML/CTF Policy (April 2021), Customer Due Diligence Standards, Sanctions Embargoes Policy, Enterprise Risk Management Framework, Records and File Retention and Destruction Policy (July 2022), CTCBL ELRA (2021, 2022), GRCC meeting minutes, Global Compliance Desk Instruction Manual, compliance registers, and logs."},
  {"question":"What types of testing and reviews were performed during this audit?","response":"Sample testing of active client files, disbursements, employee screening; review of compliance registers/logs, sanctions/Pep/SARs logs, and Sentinel screening reports."},
  {"question":"What were key noncompliance issues in CDD and risk assessment?","response":"Missing Lexis Nexis search for high-risk clients, lack of senior compliance approval for PEP-related party risk assessment, and incomplete Trigger Event Form."},
  {"question":"Which issue did the audit identify regarding staff training?","response":"A missing record of 2021 annual AML/ATF training for one existing employee; new LMS aims to prevent missed training."},
  {"question":"Did the audit find exceptions regarding beneficial owner distributions?","response":"Yes, missing signed distribution task lists and Memorandum for the 6th distribution, incorrect distribution amount; findings remediated before report issuance."},
  {"question":"Did the audit find exceptions in the employee screening process?","response":"Yes, missing employment verification for 2 new and 1 existing employee, missing education verification for 1 new employee; policy update recommended."},
  {"question":"What recommendations were made for AML/ATF policies and controls?","response":"Incorporate omitted regulatory requirements, clarify Board approval policy, update ELRA to link controls to risks, and improve documentation standards."}
]


# Create DSPy examples - for mosaic agent format
data_set = [
    # Convert to DSPy examples with mosaic agent format
    dspy.Example(
        question={'messages': [{'content': item["question"], 'role': 'user'}]},
        response=item["response"]
    ).with_inputs("question")
    for item in raw_training_data
]
example = data_set[24]
example

# COMMAND ----------

random.Random(0).shuffle(data_set)
train_dataset, test_dataset = data_set[:20], data_set[20:35]
len(train_dataset), len(test_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CREATE EVALUATION METRIC
# MAGIC - The SemanticF1 metric in DSPy evaluates long-form answers by comparing their semantic content (key ideas) to a reference answer, computing precision, recall, and their F1 harmonic mean
# MAGIC - SemanticF1 is implemented as a DSPy Module that calls an internal ChainOfThought signature to produce recall and precision, then returns F1; by default it uses a prompt that takes question, ground_truth, and system_response as inputs.
# MAGIC - Use SemanticF1 to evaluate RAG or QA systems where exact-match is too strict, and you care about capturing the important facts while avoiding unsupported content.

# COMMAND ----------

#Test in just a single row
from dspy.evaluate import SemanticF1
dspy.configure(lm=dspy.LM(LM))
metric = SemanticF1(decompositional=True)
prediction_obj = rag(**example.inputs())
score = metric(example, prediction_obj)

# Extract question text from mosaic format for display
question_text = example.question["messages"][-1]["content"]
print(f"Question: \t {question_text}\n")
print(f"Gold Response: \t {example.response}\n")
print(f"Predicted Response: \t {prediction_obj.response}\n")
print(f"Semantic F1 Score: {score:.2f}")

# COMMAND ----------

# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(devset=test_dataset, metric=metric, num_threads=24,
                         display_progress=True, display_table=2)

# Evaluate our RAG agent before any optimization 
evaluate(rag)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimize using MIPROv2

# COMMAND ----------

# Optimize the RAG agent using MIPROv2
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)
optimized_rag_v2 = tp.compile(RAG(lm_name=small_lm_name, for_mosaic_agent=True, max_history_length=max_history_length, enable_history=enable_history), trainset=train_dataset, max_bootstrapped_demos=2, max_labeled_demos=2)

# COMMAND ----------

# Evaluate the optimized RAG agent
evaluate(optimized_rag_v2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CREATE CUSTOM EVALUATION METRIC
# MAGIC We can also use custom AI Judges functions to help us with the evaluation

# COMMAND ----------

from databricks.agents.evals import judges
import numpy as np
def validate_retrieval_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Uses Databricks AI judges to validate the retrieval answer and return score (1.0 = correct, 0.0 = incorrect) plus feedback
    """
    # Extract question text from mosaic format
    if isinstance(example.question, dict) and "messages" in example.question:
        question_text = example.question["messages"][-1]["content"]
    else:
        question_text = example.question
    
    # Handle prediction which might be a string or Prediction object
    if isinstance(prediction, str):
        response_text = prediction
    else:
        response_text = prediction.response
    
    judgement = judges.correctness(
        request=question_text,
        response=response_text,
        expected_response=example.response
    )
    # obtain score from judgement
    if judgement and judgement.value:
        score = int(judgement.value == "yes")
        print(f"Judgement: {judgement.value}")
    else:
        score = int(example.response == response_text)
    
    # obtain feedback from judgement
    if judgement and judgement.rationale:
        feedback = judgement.rationale
    else:
        feedback = None
    return dspy.Prediction(score=score, feedback=feedback)

def check_accuracy(rag_agent, test_data = test_dataset):
    """
    Checks the accuracy of the agent on the test data
    """
    scores = []
    for example in test_data:
        # Use the question directly (already in mosaic format for our examples)
        prediction = rag_agent(example.question)
        score = validate_retrieval_with_feedback(example, prediction).score
        scores.append(score)
    return np.mean(scores)

# COMMAND ----------

uncompiled_rag = RAG(lm_name=small_lm_name, for_mosaic_agent=True, max_history_length=max_history_length, enable_history=enable_history)
uncompiled_small_lm_accuracy=check_accuracy(uncompiled_rag)
displayHTML(f"<h1>Uncompiled {small_lm_name} accuracy: {uncompiled_small_lm_accuracy}</h1>")

# COMMAND ----------

uncompiled_large_lm_accuracy=check_accuracy(RAG(lm_name=larger_lm_name, for_mosaic_agent=True, max_history_length=max_history_length, enable_history=enable_history))
displayHTML(f"<h1>Uncompiled {larger_lm_name} accuracy: {uncompiled_large_lm_accuracy}</h1>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log MIPROv2 Optimized Model

# COMMAND ----------

# Log the MIPROv2 optimized model to MLflow (without Unity Catalog registration)
miprov2_id = str(uuid.uuid4())
with mlflow.start_run(run_name=f"miprov2_{miprov2_id}"):
    mlflow.dspy.log_model(
        optimized_rag_v2,
        name="model"
    )
    
    # Log accuracy metrics
    miprov2_accuracy = check_accuracy(optimized_rag_v2)
    mlflow.log_metric("miprov2_accuracy", miprov2_accuracy)
    mlflow.log_metric("baseline_small_accuracy", uncompiled_small_lm_accuracy)
    mlflow.log_metric("baseline_large_accuracy", uncompiled_large_lm_accuracy)
    
    # Log key parameters
    mlflow.log_param("optimization_method", "MIPROv2")
    mlflow.log_param("small_lm_name", small_lm_name)
    mlflow.log_param("larger_lm_name", larger_lm_name)
    mlflow.log_param("reflection_lm_name", reflection_lm_name)
    mlflow.log_param("vector_search_endpoint", VECTOR_SEARCH_ENDPOINT)
    mlflow.log_param("vector_search_index", VECTOR_SEARCH_INDEX)
    mlflow.log_param("catalog", CATALOG)
    mlflow.log_param("schema", SCHEMA)
    mlflow.log_param("max_history_length", max_history_length)
    mlflow.log_param("enable_history", enable_history)
    
    print(f"MIPROv2 model logged successfully!")
    print(f"MIPROv2 accuracy: {miprov2_accuracy}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimize using GEPA

# COMMAND ----------

# defining an UUID to identify the optimized module
id = str(uuid.uuid4())
print(f"id: {id}")
#Using GEPA with Claude-Sonnet-4 to evolve the instructions based on the AI Judge feedback

gepa = dspy.GEPA(
    metric=validate_retrieval_with_feedback,
    auto="light",
    reflection_minibatch_size=5,
    reflection_lm=dspy.LM(f"databricks/{reflection_lm_name}"),
    num_threads=16,
    seed=1
)

with mlflow.start_run(run_name=f"gepa_{id}"):
    compiled_gepa = gepa.compile(
        RAG(lm_name=small_lm_name, for_mosaic_agent=True, max_history_length=max_history_length, enable_history=enable_history),
        trainset=train_dataset
    )
    
    # Log the compiled model to MLflow (without Unity Catalog registration)
    mlflow.dspy.log_model(
        compiled_gepa,
        name="model"
    )
    
    # Log accuracy metrics
    compiled_small_lm_accuracy = check_accuracy(compiled_gepa)
    mlflow.log_metric("gepa_accuracy", compiled_small_lm_accuracy)
    mlflow.log_metric("baseline_small_accuracy", uncompiled_small_lm_accuracy)
    mlflow.log_metric("baseline_large_accuracy", uncompiled_large_lm_accuracy)
    
    # Log key parameters
    mlflow.log_param("optimization_method", "GEPA")
    mlflow.log_param("small_lm_name", small_lm_name)
    mlflow.log_param("larger_lm_name", larger_lm_name)
    mlflow.log_param("reflection_lm_name", reflection_lm_name)
    mlflow.log_param("vector_search_endpoint", VECTOR_SEARCH_ENDPOINT)
    mlflow.log_param("vector_search_index", VECTOR_SEARCH_INDEX)
    mlflow.log_param("catalog", CATALOG)
    mlflow.log_param("schema", SCHEMA)
    mlflow.log_param("max_history_length", max_history_length)
    mlflow.log_param("enable_history", enable_history)
    
    print(f"GEPA model logged successfully!")
    print(f"GEPA accuracy: {compiled_small_lm_accuracy}")

compiled_gepa.save(
    "./compiled_gepa_rag_dir/",
    save_program=True
)


# COMMAND ----------

loaded_rag_gepa = dspy.load("./compiled_gepa_rag_dir/")
result_obj = loaded_rag_gepa({'messages': [{'content': 'Who is responsible for remediating the issues with the Customer Risk Assessment for the high-risk PEP client, and what is the remediation date?', 'role': 'user'}]})
result = result_obj.response if hasattr(result_obj, 'response') else result_obj

# COMMAND ----------

displayHTML(f"<h1>Compiled {small_lm_name} accuracy: {compiled_small_lm_accuracy}</h1>")