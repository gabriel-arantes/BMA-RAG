# Databricks notebook source
# MAGIC %pip install --index-url https://pypi.org/simple dspy databricks-agents mlflow[databricks]>=3.1.0 databricks-vectorsearch databricks-sdk databricks-mcp databricks-dspy uv
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # DSPy MIPROv2 Optimization Example
# MAGIC 
# MAGIC This notebook demonstrates **MIPROv2 (Multi-prompt Instruction Proposal v2)** optimization for RAG systems.
# MAGIC 
# MAGIC ## What is MIPROv2?
# MAGIC 
# MAGIC MIPROv2 is a DSPy optimizer that automatically:
# MAGIC - **Generates instructions** for your prompts through instruction evolution
# MAGIC - **Bootstraps few-shot examples** from your training data
# MAGIC - **Optimizes both** simultaneously for better performance
# MAGIC 
# MAGIC ## When to Use MIPROv2 vs GEPA
# MAGIC 
# MAGIC **Use MIPROv2 when:**
# MAGIC - You need automatic instruction generation
# MAGIC - You want few-shot examples bootstrapped from training data
# MAGIC - You have a provisioned throughput endpoint (high rate limits)
# MAGIC - You're optimizing for semantic similarity or complex metrics
# MAGIC 
# MAGIC **Use GEPA when:**
# MAGIC - You need faster iteration (less API calls)
# MAGIC - You want to compare multiple metrics
# MAGIC - You're working with pay-per-token endpoints
# MAGIC - You need production-ready optimization (used in author_model.py)
# MAGIC 
# MAGIC ## ⚠️ Important: Rate Limiting
# MAGIC 
# MAGIC MIPROv2 makes **many LLM calls** during optimization. With pay-per-token endpoints,
# MAGIC you may hit rate limits. **Recommended**: Use provisioned throughput endpoints or
# MAGIC set `num_threads=1` to minimize parallel calls.

# COMMAND ----------

# Standard library imports
import os
import random
import uuid

# Third-party imports
import dspy
import mlflow
from dspy.evaluate import SemanticF1
from dspy.retrievers.databricks_rm import DatabricksRM
from typing import Optional

# Databricks imports
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker

# MLflow imports
from mlflow.models import infer_signature
from mlflow.models.resources import DatabricksVectorSearchIndex

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration widgets
dbutils.widgets.text("experiment_name", "/Workspace/Users/garantes_oc@bma.bm/dspy_miprov2_example", "Experiment Name")
dbutils.widgets.text("catalog_name", "test_catalog", "Catalog Name")
dbutils.widgets.text("schema_name", "test_schema", "Schema Name")
dbutils.widgets.text("vector_search_endpoint", "ctcbl-unstructured-endpoint", "Vector Search Endpoint")
dbutils.widgets.text("vector_index_name", "test_volume_ctcbl_chunked_index_element__v0_0_1", "Vector Index Name")
dbutils.widgets.text("small_chat_endpoint_name", "databricks-gpt-oss-20b", "Small Chat Endpoint Name")
dbutils.widgets.text("larger_chat_endpoint_name", "databricks-claude-3-7-sonnet", "Larger Chat Endpoint Name")
dbutils.widgets.text("max_history_length", "10", "Max History Length")
dbutils.widgets.text("enable_history", "true", "Enable History")
dbutils.widgets.text("num_threads", "1", "Number of Threads for MIPROv2 (use 1 to avoid rate limits)")
dbutils.widgets.dropdown("test_mode", "true", ["true", "false"], "Quick Test Mode (3 train/3 test)")

# Get widget values
experiment_name = dbutils.widgets.get("experiment_name")
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
VECTOR_SEARCH_ENDPOINT = dbutils.widgets.get("vector_search_endpoint")
VECTOR_SEARCH_INDEX = dbutils.widgets.get("vector_index_name")
INDEX_PATH = f"{CATALOG}.{SCHEMA}.{VECTOR_SEARCH_INDEX}"

small_lm_name = dbutils.widgets.get("small_chat_endpoint_name")
larger_lm_name = dbutils.widgets.get("larger_chat_endpoint_name")
LM = f"databricks/{small_lm_name}"
max_history_length = int(dbutils.widgets.get("max_history_length"))
enable_history = dbutils.widgets.get("enable_history").lower() == "true"
num_threads = int(dbutils.widgets.get("num_threads"))
test_mode = dbutils.widgets.get("test_mode").lower() == "true"

print(f"✅ Configuration loaded:")
print(f"   Small LM: {small_lm_name}")
print(f"   Larger LM: {larger_lm_name}")
print(f"   Vector Index: {INDEX_PATH}")
print(f"   Num Threads: {num_threads}")
print(f"   Test Mode: {test_mode}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Setup

# COMMAND ----------

mlflow.set_experiment(experiment_name)
print(f"✅ MLflow experiment set: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Core Utilities
# MAGIC 
# MAGIC ### History Extraction

# COMMAND ----------

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
    
    history = dspy.History(messages=history_messages)
    
    if os.environ.get("DSPY_DEBUG_HISTORY", "false").lower() == "true":
        print(f"[DEBUG] Extracted history: {len(history_messages)} turns")
        print(f"[DEBUG] History messages: {history_messages}")
    
    return history

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Search Tool

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

        self.vs_client = VectorSearchClient()
        self.index = self.vs_client.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )

    def search(self, query: str) -> str:
        """Perform vector search and return formatted results as a string."""
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### DSPy Signature

# COMMAND ----------

class BMAChatAssistant(dspy.Signature):
    """
    You are a trusted assistant that helps answer questions based only on the provided information.
    """
    context: str = dspy.InputField(desc="The retrieved or provided context to answer the customer's question.")
    question: str = dspy.InputField(desc="The customer's question that needs to be answered.")
    history: dspy.History = dspy.InputField(desc="A record of previous conversation turns as question/response pairs.")
    response: str = dspy.OutputField(desc="The assistant's answer to the customer's question, based solely on the context.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAG Module

# COMMAND ----------

class RAG(dspy.Module):
    def __init__(self, lm_name, for_mosaic_agent=True, max_history_length=None, enable_history=True):
        mlflow.dspy.autolog()

        self.for_mosaic_agent = for_mosaic_agent
        self.lm = dspy.LM(model=f"databricks/{lm_name}")
        self.max_history_length = max_history_length if max_history_length is not None else globals().get('max_history_length', 10)
        self.enable_history = enable_history if enable_history is not None else globals().get('enable_history', True)

        self.retriever = DatabricksRM(
            databricks_index_name=INDEX_PATH,
            text_column_name="chunk_content",
            docs_id_column_name="chunk_id",
            columns=["chunk_id", "chunk_content", "path"],
            k=5,
            use_with_databricks_agent_framework=for_mosaic_agent,
        )

        self.response_generator = dspy.Predict(BMAChatAssistant)

    def forward(self, question):
        if self.for_mosaic_agent:
            if isinstance(question, dict) and "messages" in question:
                question_text = question["messages"][-1]["content"]
                history = extract_history_from_messages(question["messages"], self.max_history_length, self.enable_history)
            else:
                question_text = question
                history = dspy.History(messages=[])
        else:
            if isinstance(question, dict) and "messages" in question:
                question_text = question["messages"][-1]["content"] if question["messages"] else str(question)
                history = extract_history_from_messages(question["messages"], self.max_history_length, self.enable_history)
            else:
                question_text = question
                history = dspy.History(messages=[])

        context = self.retriever(question_text)

        with dspy.context(lm=self.lm):
            response = self.response_generator(context=context, question=question_text, history=history)
            return response

print("✅ RAG class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Data

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
    dspy.Example(
        question={'messages': [{'content': item["question"], 'role': 'user'}]},
        response=item["response"]
    ).with_inputs("question")
    for item in raw_training_data
]

print(f"✅ Created {len(data_set)} training examples")

# COMMAND ----------

# Shuffle and split data
random.Random(0).shuffle(data_set)

if test_mode:
    train_dataset, test_dataset = data_set[:3], data_set[3:6]
    print("🧪 TEST MODE: Using 3 train / 3 test examples for quick testing")
else:
    split_idx = int(len(data_set) * 0.7)
    train_dataset, test_dataset = data_set[:split_idx], data_set[split_idx:]
    print(f"📊 FULL MODE: Using {len(train_dataset)} train / {len(test_dataset)} test examples (all available data)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metric: SemanticF1

# COMMAND ----------

dspy.configure(lm=dspy.LM(LM))
semantic_f1_metric = SemanticF1(decompositional=True)

# Test on a single example
rag = RAG(lm_name=small_lm_name, for_mosaic_agent=True, max_history_length=max_history_length, enable_history=enable_history)
example = test_dataset[0]
prediction_obj = rag(**example.inputs())
score = semantic_f1_metric(example, prediction_obj)

question_text = example.question["messages"][-1]["content"]
print(f"✅ SemanticF1 metric configured")
print(f"\nSample evaluation:")
print(f"Question: {question_text}")
print(f"Gold Response: {example.response}")
print(f"Predicted Response: {prediction_obj.response}")
print(f"Semantic F1 Score: {score:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Function

# COMMAND ----------

def check_accuracy(rag_agent, test_data=None):
    """
    Evaluate agent accuracy using SemanticF1 metric.
    
    Args:
        rag_agent: The RAG agent to evaluate
        test_data: Test dataset (defaults to global test_dataset)
        
    Returns:
        float: Mean accuracy score (as decimal, e.g. 0.6730) for .2% formatting
    """
    from dspy.evaluate import Evaluate, SemanticF1
    
    data = test_data if test_data is not None else globals()['test_dataset']
    metric_fn = SemanticF1(decompositional=True)
    
    evaluator = Evaluate(
        devset=data,
        metric=metric_fn,
        num_threads=num_threads,
        display_progress=True,
        display_table=False
    )
    
    result = evaluator(rag_agent)
    return result.score / 100.0

print("✅ Evaluation function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Evaluation

# COMMAND ----------

print("="*80)
print("📊 BASELINE EVALUATION")
print("="*80)

# Evaluate uncompiled RAG with small LM
print(f"\n🔹 Evaluating uncompiled {small_lm_name}...")
uncompiled_rag = RAG(lm_name=small_lm_name, for_mosaic_agent=True, max_history_length=max_history_length, enable_history=enable_history)
uncompiled_small_lm_accuracy = check_accuracy(uncompiled_rag)
print(f"✅ Uncompiled {small_lm_name} accuracy: {uncompiled_small_lm_accuracy:.2%}")

# Evaluate uncompiled RAG with larger LM
print(f"\n🔹 Evaluating uncompiled {larger_lm_name}...")
uncompiled_large_lm_accuracy = check_accuracy(RAG(lm_name=larger_lm_name, for_mosaic_agent=True, max_history_length=max_history_length, enable_history=enable_history))
print(f"✅ Uncompiled {larger_lm_name} accuracy: {uncompiled_large_lm_accuracy:.2%}")

print("\n" + "="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MIPROv2 Optimization
# MAGIC 
# MAGIC **What MIPROv2 Does:**
# MAGIC - Automatically generates optimized instructions for your prompts
# MAGIC - Bootstraps few-shot examples from training data
# MAGIC - Evolves instructions through multiple iterations
# MAGIC - Optimizes for the specified metric (SemanticF1)
# MAGIC 
# MAGIC **Parameters:**
# MAGIC - `auto="medium"`: Automatic configuration (light/medium/heavy)
# MAGIC - `num_threads`: Parallel evaluation threads (use 1 for rate limit safety)
# MAGIC - `max_bootstrapped_demos`: Maximum bootstrapped examples per prompt
# MAGIC - `max_labeled_demos`: Maximum labeled examples per prompt

# COMMAND ----------

print("="*80)
print("🔧 MIPROV2 OPTIMIZATION")
print("="*80)

miprov2_id = str(uuid.uuid4())
print(f"Run ID: {miprov2_id}")
print(f"Num Threads: {num_threads}")
print(f"Metric: SemanticF1 (decompositional=True)")
print("="*80)

# Initialize MIPROv2 optimizer
tp = dspy.MIPROv2(metric=semantic_f1_metric, auto="medium", num_threads=num_threads)

# Start MLflow run for tracking
with mlflow.start_run(run_name=f"miprov2_{miprov2_id}"):
    print("\n🔄 Starting MIPROv2 compilation...")
    print("   This may take several minutes...")
    
    # Compile the RAG model with MIPROv2
    optimized_rag_v2 = tp.compile(
        RAG(lm_name=small_lm_name, for_mosaic_agent=True, max_history_length=max_history_length, enable_history=enable_history),
        trainset=train_dataset,
        max_bootstrapped_demos=2,
        max_labeled_demos=2
    )
    
    print("✅ MIPROv2 compilation complete!")
    
    # Evaluate optimized model
    print("\n📊 Evaluating MIPROv2-optimized model...")
    miprov2_accuracy = check_accuracy(optimized_rag_v2)
    
    # Log metrics
    mlflow.log_metric("miprov2_accuracy", miprov2_accuracy)
    mlflow.log_metric("baseline_small_accuracy", uncompiled_small_lm_accuracy)
    mlflow.log_metric("baseline_large_accuracy", uncompiled_large_lm_accuracy)
    mlflow.log_metric("improvement_over_small", miprov2_accuracy - uncompiled_small_lm_accuracy)
    mlflow.log_metric("improvement_over_large", miprov2_accuracy - uncompiled_large_lm_accuracy)
    
    # Log parameters
    mlflow.log_param("optimization_method", "MIPROv2")
    mlflow.log_param("small_lm_name", small_lm_name)
    mlflow.log_param("larger_lm_name", larger_lm_name)
    mlflow.log_param("vector_search_endpoint", VECTOR_SEARCH_ENDPOINT)
    mlflow.log_param("vector_search_index", VECTOR_SEARCH_INDEX)
    mlflow.log_param("catalog", CATALOG)
    mlflow.log_param("schema", SCHEMA)
    mlflow.log_param("num_threads", num_threads)
    mlflow.log_param("max_history_length", max_history_length)
    mlflow.log_param("enable_history", enable_history)
    mlflow.log_param("test_mode", test_mode)
    mlflow.log_param("max_bootstrapped_demos", 2)
    mlflow.log_param("max_labeled_demos", 2)
    mlflow.log_param("auto_mode", "medium")
    
    # Log the optimized model
    from mlflow.models import infer_signature
    from mlflow.models.resources import DatabricksVectorSearchIndex
    
    input_example = {
        'messages': [
            {'content': 'What was the overall audit rating for CTCBL?', 'role': 'user'}
        ]
    }
    
    signature_prediction = optimized_rag_v2(input_example)
    signature = infer_signature(input_example, signature_prediction)
    
    mlflow.dspy.log_model(
        optimized_rag_v2,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        resources=[
            DatabricksVectorSearchIndex(index_name=INDEX_PATH)
        ],
        pip_requirements=[
            "dspy",
            "mlflow[databricks]>=3.1.0",
            "databricks-vectorsearch",
            "databricks-sdk",
            "databricks-dspy"
        ]
    )
    
    print(f"\n✅ MIPROv2 model logged successfully!")
    print(f"   MIPROv2 accuracy: {miprov2_accuracy:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

print("="*80)
print("🎉 OPTIMIZATION COMPLETE")
print("="*80)

print(f"\n📊 Results Summary:")
print(f"   Baseline (Small LM):  {uncompiled_small_lm_accuracy:.2%}")
print(f"   Baseline (Large LM):  {uncompiled_large_lm_accuracy:.2%}")
print(f"   MIPROv2 Optimized:    {miprov2_accuracy:.2%}")

print(f"\n📈 Improvements:")
print(f"   vs Small LM: {(miprov2_accuracy - uncompiled_small_lm_accuracy):.2%}")
print(f"   vs Large LM: {(miprov2_accuracy - uncompiled_large_lm_accuracy):.2%}")

if miprov2_accuracy > uncompiled_small_lm_accuracy:
    print(f"\n🏆 MIPROv2 improved over small baseline!")
else:
    print(f"\n⚠️  MIPROv2 did not improve over baseline (may need more training data or different configuration)")

print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Results

# COMMAND ----------

# Determine winner
results = [
    ("Uncompiled Small LM", uncompiled_small_lm_accuracy),
    ("Uncompiled Large LM", uncompiled_large_lm_accuracy),
    ("MIPROv2 Optimized", miprov2_accuracy)
]
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
winner = sorted_results[0][0]

displayHTML(f"""
<div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f0f8ff; border-radius: 10px; border: 2px solid #4CAF50;">
    <h1 style="color: #2e7d32;">✅ MIPROv2 Optimization Complete</h1>
    <hr style="border: 1px solid #4CAF50;">
    
    <h2 style="color: #1565c0;">📊 Performance Comparison</h2>
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <tr style="background-color: #e3f2fd;">
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Model</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Accuracy</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Status</th>
        </tr>
        {''.join([
            f'''<tr style="background-color: {'#e8f5e9' if name == winner else 'white'};">
                <td style="padding: 10px; border: 1px solid #90caf9;">{name}</td>
                <td style="padding: 10px; border: 1px solid #90caf9;">{acc:.2%}</td>
                <td style="padding: 10px; border: 1px solid #90caf9;">{'🏆 Best' if name == winner else ''}</td>
            </tr>'''
            for name, acc in sorted_results
        ])}
    </table>
    
    <h2 style="color: #1565c0;">💡 Key Insights</h2>
    <ul style="font-size: 16px; line-height: 1.8;">
        <li><strong>MIPROv2 Optimization:</strong> Automatic instruction generation and few-shot bootstrapping</li>
        <li><strong>Metric Used:</strong> SemanticF1 (decompositional)</li>
        <li><strong>Training Data:</strong> {len(train_dataset)} examples</li>
        <li><strong>Test Data:</strong> {len(test_dataset)} examples</li>
        <li><strong>Improvement over Small LM:</strong> {(miprov2_accuracy - uncompiled_small_lm_accuracy):.2%}</li>
        <li><strong>Improvement over Large LM:</strong> {(miprov2_accuracy - uncompiled_large_lm_accuracy):.2%}</li>
    </ul>
    
    <h2 style="color: #1565c0;">🎯 Next Steps</h2>
    <ul style="font-size: 16px; line-height: 1.8;">
        <li>Experiment with different <code>auto</code> settings (light/medium/heavy)</li>
        <li>Adjust <code>max_bootstrapped_demos</code> and <code>max_labeled_demos</code></li>
        <li>Try with provisioned throughput endpoints for higher <code>num_threads</code></li>
        <li>Compare with GEPA optimization (see author_model.py)</li>
        <li>Deploy the best model to Model Serving for production use</li>
    </ul>
    
    <div style="margin-top: 20px; padding: 15px; background-color: #fff3e0; border-left: 4px solid #ff9800;">
        <strong>⚠️ Note:</strong> If you encountered rate limiting errors, try setting <code>num_threads=1</code> or use a provisioned throughput endpoint.
    </div>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook demonstrated:
# MAGIC 
# MAGIC 1. ✅ **MIPROv2 Optimization**: Automatic instruction and few-shot example generation
# MAGIC 2. ✅ **SemanticF1 Evaluation**: Semantic similarity metric for long-form answers
# MAGIC 3. ✅ **Baseline Comparison**: Evaluated against uncompiled small and large LMs
# MAGIC 4. ✅ **MLflow Integration**: Full experiment tracking and model logging
# MAGIC 5. ✅ **Test Mode**: Quick iteration with small datasets
# MAGIC 
# MAGIC ### MIPROv2 vs GEPA
# MAGIC 
# MAGIC - **MIPROv2**: Better for semantic tasks, needs more API calls, automatic instruction generation
# MAGIC - **GEPA**: Faster, better for production, supports metric comparison, more robust to rate limits
# MAGIC 
# MAGIC ### Key Takeaways
# MAGIC 
# MAGIC - MIPROv2 can significantly improve model performance through instruction optimization
# MAGIC - Rate limiting is a consideration with pay-per-token endpoints
# MAGIC - Few-shot examples help the model understand the task better
# MAGIC - Compare with GEPA to see which optimizer works best for your use case

