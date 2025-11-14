# Databricks notebook source
# MAGIC %pip install --index-url https://pypi.org/simple dspy databricks-agents mlflow[databricks]>=3.1.0 databricks-vectorsearch databricks-sdk databricks-mcp databricks-dspy uv
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Standard library imports
import datetime
import os
import random
import uuid

# Third-party imports
import dspy
import mlflow
import numpy as np
from dspy.evaluate import Evaluate, SemanticF1
from dspy.retrievers.databricks_rm import DatabricksRM
from typing import Optional

# Databricks imports
from databricks.agents.evals import judges
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker

# MLflow imports
from mlflow.models import infer_signature
from mlflow.models.resources import DatabricksVectorSearchIndex

# COMMAND ----------

# MAGIC %md
# MAGIC # AUTHOR AGENT USING DSPY

# COMMAND ----------

# Get parameters from widgets
dbutils.widgets.text("experiment_name", "/Workspace/Users/garantes_oc@bma.bm/bma_author_dspy_model_experiment", "Experiment Name")
dbutils.widgets.text("catalog_name", "test_catalog", "Catalog Name")
dbutils.widgets.text("schema_name", "test_schema", "Schema Name")
dbutils.widgets.text("model_name", "bma_dspy_model", "Model Name")
dbutils.widgets.text("vector_search_endpoint", "ctcbl-unstructured-endpoint", "Vector Search Endpoint")
dbutils.widgets.text("vector_index_name", "test_volume_ctcbl_chunked_index_element__v0_0_1", "Vector Index Name")
dbutils.widgets.text("evaluation_dataset_table", "rag_eval_dataset", "Evaluation Dataset Table")
dbutils.widgets.text("chat_endpoint_name", "databricks-claude-3-7-sonnet", "Chat Endpoint Name")
dbutils.widgets.text("small_chat_endpoint_name", "databricks-gpt-oss-20b", "Small Chat Endpoint Name")
dbutils.widgets.text("larger_chat_endpoint_name", "databricks-claude-3-7-sonnet", "Larger Chat Endpoint Name")
dbutils.widgets.text("reflection_chat_endpoint_name", "databricks-claude-sonnet-4-5", "Reflection Chat Endpoint Name")
dbutils.widgets.text("max_history_length", "10", "Max History Length")
dbutils.widgets.text("enable_history", "true", "Enable History")
dbutils.widgets.text("num_threads", "2", "Number of Threads for GEPA Optimization")
dbutils.widgets.dropdown("test_mode", "true", ["true", "false"], "Quick Test Mode (3 train/3 test)")

# Get widget values
experiment_name = dbutils.widgets.get("experiment_name")
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
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
num_threads = int(dbutils.widgets.get("num_threads"))
test_mode = dbutils.widgets.get("test_mode").lower() == "true"

# Full model name for Unity Catalog registration
FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Experiment Setup

# COMMAND ----------

# Set up MLflow experiment and Unity Catalog
mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri("databricks-uc")

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

# MAGIC %md
# MAGIC ## Create an instance of the agent and vibe check

# COMMAND ----------

from dspy_program import RAG, BMAChatAssistant, extract_history_from_messages

rag = RAG(
    lm_name=small_lm_name,
    index_path=INDEX_PATH,
    max_history_length=max_history_length,
    enable_history=enable_history
)

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
        messages={'messages': [{'content': item["question"], 'role': 'user'}]},
        question=item["question"],
        response=item["response"]
    ).with_inputs("messages")
    for item in raw_training_data
]
example = data_set[24]
example

# COMMAND ----------

random.Random(0).shuffle(data_set)

# Use tiny samples in test mode for quick iteration, or all available data in full mode
if test_mode:
    train_dataset, test_dataset = data_set[:3], data_set[3:6]
    print("🧪 TEST MODE: Using 3 train / 3 test examples for quick testing")
else:
    # Full mode: Use ~70% for train, ~30% for test (all available data)
    split_idx = int(len(data_set) * 0.7)
    train_dataset, test_dataset = data_set[:split_idx], data_set[split_idx:]
    print(f"📊 FULL MODE: Using {len(train_dataset)} train / {len(test_dataset)} test examples (all available data)")

len(train_dataset), len(test_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CREATE EVALUATION METRIC
# MAGIC - The SemanticF1 metric in DSPy evaluates long-form answers by comparing their semantic content (key ideas) to a reference answer, computing precision, recall, and their F1 harmonic mean
# MAGIC - SemanticF1 is implemented as a DSPy Module that calls an internal ChainOfThought signature to produce recall and precision, then returns F1; by default it uses a prompt that takes question, ground_truth, and system_response as inputs.
# MAGIC - Use SemanticF1 to evaluate RAG or QA systems where exact-match is too strict, and you care about capturing the important facts while avoiding unsupported content.

# COMMAND ----------

# Test in just a single row
dspy.configure(lm=dspy.LM(LM))
metric = SemanticF1(decompositional=True)
prediction_obj = rag(**example.inputs())
score = metric(example, prediction_obj)

# Extract question text from mosaic format for display
question_text = example.question
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
# MAGIC ## CREATE CUSTOM EVALUATION METRIC
# MAGIC We can also use custom AI Judges functions to help us with the evaluation

# COMMAND ----------

def validate_retrieval_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Uses Databricks AI judges to validate the retrieval answer and return score (1.0 = correct, 0.0 = incorrect) plus feedback
    """

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

def check_accuracy(rag_agent, test_data=None, metric="custom"):
    """
    Evaluate agent accuracy using DSPy's parallel evaluation.
    
    Args:
        rag_agent: The RAG agent to evaluate
        test_data: Test dataset (defaults to global test_dataset)
        metric: "custom" for AI Judge, "semanticf1" for SemanticF1
        
    Returns:
        float: Mean accuracy score (as decimal, e.g. 0.6730) for .2% formatting
    """
    # Use global test_dataset if not provided
    data = test_data if test_data is not None else globals()['test_dataset']
    
    # Select evaluation metric
    metric_fn = (SemanticF1(decompositional=True) 
                 if metric == "semanticf1" 
                 else validate_retrieval_with_feedback)
    
    # Use DSPy's parallel evaluator for efficiency
    evaluator = Evaluate(
        devset=data,
        metric=metric_fn,
        num_threads=num_threads,
        display_progress=True,
        display_table=False
    )
    
    # Return the score attribute from EvaluationResult (percentage float, e.g. 67.30)
    result = evaluator(rag_agent)
    return result.score / 100.0  # Convert percentage (67.30) to decimal (0.6730) for formatting

# COMMAND ----------

uncompiled_rag = RAG(
    lm_name=small_lm_name,
    index_path=INDEX_PATH,
    max_history_length=max_history_length,
    enable_history=enable_history,
    for_mosaic_agent=True
)
uncompiled_small_lm_accuracy=check_accuracy(uncompiled_rag)
displayHTML(f"<h1>Uncompiled {small_lm_name} accuracy: {uncompiled_small_lm_accuracy}</h1>")

# COMMAND ----------

uncompiled_large_lm_accuracy=check_accuracy(RAG(
    lm_name=larger_lm_name,
    index_path=INDEX_PATH,
    max_history_length=max_history_length,
    enable_history=enable_history,
    for_mosaic_agent=True
))
displayHTML(f"<h1>Uncompiled {larger_lm_name} accuracy: {uncompiled_large_lm_accuracy}</h1>")

# COMMAND ----------

# MAGIC %md
# MAGIC ### GEPA Optimization with Metric Comparison
# MAGIC
# MAGIC We train two GEPA models with different metrics and select the best:
# MAGIC - **GEPA #1**: Custom AI Judge (Correctness)
# MAGIC - **GEPA #2**: SemanticF1
# MAGIC
# MAGIC The best performing model is registered to Unity Catalog.

# COMMAND ----------

# Importar o wrapper e libs adicionais
import json
from mlflow.tracking import MlflowClient

# Generate unique experiment group ID for linking all related runs
experiment_group_id = str(uuid.uuid4())
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Define input example once for all runs (DRY principle)
# Format matches what forward(messages) expects: messages = {'messages': [...]}
input_example = {
    'messages': [
        {'content': 'What was the overall audit rating for CTCBL?', 'role': 'user'}
    ]
}

print("="*80)
print(f"🚀 STARTING MODEL TRAINING COMPARISON")
print(f"Experiment Group ID: {experiment_group_id}")
print("="*80)

# ==================================================
# GEPA TRAINING #1: Custom AI Judge
# ==================================================
id_custom = str(uuid.uuid4())
print(f"\n{'='*80}")
print(f"📊 GEPA TRAINING #1: Custom AI Judge")
print(f"Run ID: {id_custom}")
print(f"{'='*80}")

gepa_custom = dspy.GEPA(
    metric=validate_retrieval_with_feedback,
    #auto="light",
    max_full_evals= 2,
    reflection_minibatch_size=5,
    reflection_lm=dspy.LM(f"databricks/{reflection_lm_name}"),
    num_threads=num_threads,
    seed=1
)

with mlflow.start_run(run_name=f"gepa_custom_{id_custom}") as run_custom:
    # Add tags for linking and traceability
    mlflow.set_tag("experiment_group", experiment_group_id)
    mlflow.set_tag("optimization_type", "gepa_custom")
    mlflow.set_tag("training_metric", "custom_ai_judge_correctness")
    mlflow.set_tag("role", "training")
    
    # Importar a classe RAG do arquivo .py
    from dspy_program import RAG
    
    compiled_gepa_custom = gepa_custom.compile(
        RAG(
            lm_name=small_lm_name,
            index_path=INDEX_PATH,
            max_history_length=max_history_length,
            enable_history=enable_history,
            for_mosaic_agent=True
        ),
        trainset=train_dataset
    )
    
    # Evaluate with Custom AI Judge metric
    gepa_custom_accuracy = check_accuracy(compiled_gepa_custom, metric="custom")
    mlflow.log_metric("gepa_custom_accuracy", gepa_custom_accuracy)
    mlflow.log_metric("baseline_small_accuracy", uncompiled_small_lm_accuracy)
    mlflow.log_metric("baseline_large_accuracy", uncompiled_large_lm_accuracy)
    
    # Log parameters
    mlflow.log_param("optimization_method", "GEPA")
    mlflow.log_param("training_metric", "custom_ai_judge_correctness")
    mlflow.log_param("small_lm_name", small_lm_name)
    mlflow.log_param("larger_lm_name", larger_lm_name)
    mlflow.log_param("reflection_lm_name", reflection_lm_name)
    mlflow.log_param("vector_search_endpoint", VECTOR_SEARCH_ENDPOINT)
    mlflow.log_param("vector_search_index", VECTOR_SEARCH_INDEX)
    mlflow.log_param("catalog", CATALOG)
    mlflow.log_param("schema", SCHEMA)
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("max_history_length", max_history_length)
    mlflow.log_param("enable_history", enable_history)
    mlflow.log_param("num_threads", num_threads)
    mlflow.log_param("experiment_group", experiment_group_id)
    
    # Log the trained model with full UC metadata (signature, input_example)
    signature_prediction = compiled_gepa_custom(input_example)
    signature = infer_signature(input_example, signature_prediction)
    
    # 1. Salve o modelo DSPy treinado (prompts) localmente
    model_save_path = "./compiled_gepa_custom_dir"
    compiled_gepa_custom.save(model_save_path, save_program=True)
    print(f"Modelo DSPy compilado salvo em: {model_save_path}")

    # 2. Crie um dict de configuração com os parâmetros de __init__
    config = {
        "lm_name": small_lm_name,
        "index_path": INDEX_PATH,
        "max_history_length": max_history_length,
        "enable_history": enable_history
    }
    
    # 3. Salve a configuração em um diretório para artefatos
    config_dir = "./config_custom" # Nome único para este run
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # 4. Defina os requisitos
    pip_requirements = [
        "dspy",
        "mlflow[databricks]>=3.1.0",
        "databricks-vectorsearch",
        "databricks-sdk",
        "databricks-dspy",
        "pandas" 
    ]

    mlflow.dspy.log_model(
        dspy_model=compiled_gepa_custom,
        name="model",
        code_paths=["dspy_program.py"],
        signature=signature,
        input_example=input_example,
        resources=[
            DatabricksVectorSearchIndex(index_name=INDEX_PATH)
        ],
        pip_requirements=pip_requirements
    )
    
    print(f"✅ GEPA (Custom AI Judge) model trained and LOGGED WITH PYFUNC")
    print(f"   Accuracy: {gepa_custom_accuracy:.2%}")

id_semantic = str(uuid.uuid4())
print(f"\n{'='*80}")
print(f"📊 GEPA TRAINING #2: SemanticF1")
print(f"Run ID: {id_semantic}")
print(f"{'='*80}")

semantic_f1_metric = SemanticF1(decompositional=True)

def wrapped_semantic_f1(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Wraps SemanticF1 to accept extra GEPA reflection args and ignore them."""
    return semantic_f1_metric(example, prediction, trace=trace)


gepa_semantic = dspy.GEPA(
    metric=wrapped_semantic_f1, 
    #auto="light",
    max_full_evals= 2,
    reflection_minibatch_size=5,
    reflection_lm=dspy.LM(f"databricks/{reflection_lm_name}"),
    num_threads=num_threads,
    seed=1
)

with mlflow.start_run(run_name=f"gepa_semantic_{id_semantic}") as run_semantic:
    # Add tags for linking and traceability
    mlflow.set_tag("experiment_group", experiment_group_id)
    mlflow.set_tag("optimization_type", "gepa_semantic")
    mlflow.set_tag("training_metric", "semantic_f1")
    mlflow.set_tag("role", "training")
    
    # Importar a classe RAG do arquivo .py
    from dspy_program import RAG
    
    compiled_gepa_semantic = gepa_semantic.compile(
        RAG(
            lm_name=small_lm_name,
            index_path=INDEX_PATH,
            max_history_length=max_history_length,
            enable_history=enable_history,
            for_mosaic_agent=True
        ),
        trainset=train_dataset
    )
    
    # Evaluate with SemanticF1 metric
    gepa_semantic_accuracy = check_accuracy(compiled_gepa_semantic, metric="semanticf1")
    mlflow.log_metric("gepa_semantic_accuracy", gepa_semantic_accuracy)
    mlflow.log_metric("baseline_small_accuracy", uncompiled_small_lm_accuracy)
    mlflow.log_metric("baseline_large_accuracy", uncompiled_large_lm_accuracy)
    
    # Log parameters
    mlflow.log_param("optimization_method", "GEPA")
    mlflow.log_param("training_metric", "semantic_f1")
    mlflow.log_param("small_lm_name", small_lm_name)
    mlflow.log_param("larger_lm_name", larger_lm_name)
    mlflow.log_param("reflection_lm_name", reflection_lm_name)
    mlflow.log_param("vector_search_endpoint", VECTOR_SEARCH_ENDPOINT)
    mlflow.log_param("vector_search_index", VECTOR_SEARCH_INDEX)
    mlflow.log_param("catalog", CATALOG)
    mlflow.log_param("schema", SCHEMA)
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("max_history_length", max_history_length)
    mlflow.log_param("enable_history", enable_history)
    mlflow.log_param("num_threads", num_threads)
    mlflow.log_param("experiment_group", experiment_group_id)
    
    # Log the trained model with full UC metadata (signature, input_example)
    signature_prediction = compiled_gepa_semantic(input_example)
    signature = infer_signature(input_example, signature_prediction)

    # 1. Salve o modelo DSPy treinado (prompts) localmente
    model_save_path = "./compiled_gepa_semantic_dir"
    compiled_gepa_semantic.save(model_save_path, save_program=True)
    print(f"Modelo DSPy compilado salvo em: {model_save_path}")

    # 2. Crie um dict de configuração com os parâmetros de __init__
    config = {
        "lm_name": small_lm_name,
        "index_path": INDEX_PATH,
        "max_history_length": max_history_length,
        "enable_history": enable_history
    }
    
    # 3. Salve a configuração em um diretório para artefatos
    config_dir = "./config_semantic" # Nome único para este run
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)

    # 4. Defina os requisitos
    pip_requirements = [
        "dspy",
        "mlflow[databricks]>=3.1.0",
        "databricks-vectorsearch",
        "databricks-sdk",
        "databricks-dspy",
        "pandas" 
    ]

    mlflow.dspy.log_model(
        dspy_model=compiled_gepa_semantic,
        name="model",
        code_paths=["dspy_program.py"], 
        signature=signature,
        input_example=input_example,
        resources=[
            DatabricksVectorSearchIndex(index_name=INDEX_PATH)
        ],
        pip_requirements=pip_requirements
    )
    
    print(f"✅ GEPA (SemanticF1) model trained and LOGGED WITH PYFUNC")
    print(f"   Accuracy: {gepa_semantic_accuracy:.2%}")


# ==================================================
# RUN #3: COMPARE METRICS AND REGISTER BEST MODEL
# ==================================================
print(f"\n{'='*80}")
print("⚖️  COMPARING METRICS AND REGISTERING BEST MODEL")
print(f"{'='*80}")

with mlflow.start_run(run_name=f"model_comparison_{timestamp}") as comparison_run:
    # Add tags for linking
    mlflow.set_tag("experiment_group", experiment_group_id)
    mlflow.set_tag("role", "comparison_and_registration")
    
    # Collect all model results in a dictionary list (scalable approach)
    model_results = [
        {
            "name": "Custom AI Judge",
            "metric_name": "custom_ai_judge_correctness",
            "model": compiled_gepa_custom, 
            "accuracy": gepa_custom_accuracy,
            "run_id": run_custom.info.run_id
        },
        {
            "name": "SemanticF1",
            "metric_name": "semantic_f1",
            "model": compiled_gepa_semantic, 
            "accuracy": gepa_semantic_accuracy,
            "run_id": run_semantic.info.run_id
        }
    ]
    
    # Select best model using max() - scalable and elegant!
    best_result = max(model_results, key=lambda x: x["accuracy"])
    
    # Extract values
    best_model = best_result["model"] 
    best_metric_name = best_result["metric_name"]
    best_accuracy = best_result["accuracy"]
    best_run_id = best_result["run_id"]
    winner = best_result["name"]
    
    # Get runner-up and all metrics compared
    sorted_results = sorted(model_results, key=lambda x: x["accuracy"], reverse=True)
    runner_up_accuracy = sorted_results[1]["accuracy"] if len(sorted_results) > 1 else 0.0
    all_metrics_compared = ",".join([r["metric_name"] for r in model_results])
    
    # Print comparison results
    print(f"\nModel Comparison Results:")
    for i, result in enumerate(sorted_results, 1):
        marker = "🏆" if result["name"] == winner else f"  {i}."
        print(f"{marker} {result['name']:20s} - {result['accuracy']:.2%}")
    
    print(f"\n✅ Best Model: {winner}")
    print(f"   Accuracy: {best_accuracy:.2%}")
    print(f"   Improvement: {(best_accuracy - runner_up_accuracy):.2%} over runner-up")
    
    # Log comparison metrics
    mlflow.log_metric("best_model_accuracy", best_accuracy)
    mlflow.log_metric("runner_up_accuracy", runner_up_accuracy)
    mlflow.log_metric("accuracy_improvement", best_accuracy - runner_up_accuracy)
    mlflow.log_metric("baseline_small_accuracy", uncompiled_small_lm_accuracy)
    mlflow.log_metric("baseline_large_accuracy", uncompiled_large_lm_accuracy)
    
    # Log comparison parameters
    mlflow.log_param("winning_metric", best_metric_name)
    mlflow.log_param("best_training_run_id", best_run_id)
    mlflow.log_param("experiment_group", experiment_group_id)
    mlflow.log_param("comparison_performed", "yes")
    mlflow.log_param("num_models_compared", len(model_results))
    
    # Log all training run details
    for i, result in enumerate(model_results):
        mlflow.log_param(f"training_run_{i+1}_id", result["run_id"])
        mlflow.log_param(f"training_run_{i+1}_name", result["name"])
        mlflow.log_param(f"training_run_{i+1}_metric", result["metric_name"])
        mlflow.log_param(f"training_run_{i+1}_accuracy", result["accuracy"])
    
    # ==================================================
    # REGISTER BEST MODEL FROM TRAINING RUN
    # ==================================================
    print(f"\n{'='*80}")
    print("📦 REGISTERING BEST MODEL TO UNITY CATALOG")
    print(f"{'='*80}")
    print(f"Registering from training run: {best_run_id}")
    
    # Register model using URI from best training run (no re-logging)
    model_uri = f"runs:/{best_run_id}/model"
    
    client = MlflowClient()
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=FULL_MODEL_NAME
    )
    
    # Get model version info
    model_info = client.get_model_version(
        name=FULL_MODEL_NAME,
        version=model_version.version
    )
    
    print(f"✅ Model registered to Unity Catalog")
    print(f"   Model: {FULL_MODEL_NAME}")
    print(f"   Version: {model_version.version}")
    print(f"   Source Run: {best_run_id}")
# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Model Alias and Tags

# COMMAND ----------

# ==================================================
# SET MODEL ALIAS AND TAGS
# ==================================================
print(f"{'='*80}")
print("🏷️  SETTING MODEL ALIAS AND TAGS")
print(f"{'='*80}")

client = mlflow.tracking.MlflowClient()

# Update model description
client.update_registered_model(
    name=FULL_MODEL_NAME,
    description=f"BMA Author DSPy Model - GEPA optimized RAG system trained with {winner} metric (selected via metric comparison)"
)

# Set registered model tags
client.set_registered_model_tag(FULL_MODEL_NAME, "team", "bma_dsai_mde")
client.set_registered_model_tag(FULL_MODEL_NAME, "use_case", "document_qa")
client.set_registered_model_tag(FULL_MODEL_NAME, "optimization_method", "GEPA")
client.set_registered_model_tag(FULL_MODEL_NAME, "metrics_compared", all_metrics_compared)
client.set_registered_model_tag(FULL_MODEL_NAME, "metric_comparison", "yes")
client.set_registered_model_tag(FULL_MODEL_NAME, "winning_metric", best_metric_name)

# Set version tags for traceability
client.set_model_version_tag(
    name=FULL_MODEL_NAME,
    version=model_version.version,
    key="best_training_run_id",
    value=best_run_id  # ← Direct link to winning training run!
)

client.set_model_version_tag(
    name=FULL_MODEL_NAME,
    version=model_version.version,
    key="comparison_run_id",
    value=comparison_run.info.run_id
)

client.set_model_version_tag(
    name=FULL_MODEL_NAME,
    version=model_version.version,
    key="experiment_group",
    value=experiment_group_id
)

client.set_model_version_tag(
    name=FULL_MODEL_NAME,
    version=model_version.version,
    key="validation_status",
    value="pending"
)

# Log all training run IDs for reference
all_training_run_ids = ",".join([r["run_id"] for r in model_results])
client.set_model_version_tag(
    name=FULL_MODEL_NAME,
    version=model_version.version,
    key="all_training_runs",
    value=all_training_run_ids
)

# Set challenger alias
client.set_registered_model_alias(
    name=FULL_MODEL_NAME,
    alias="challenger",
    version=model_version.version
)

print(f"✅ Model alias set: challenger")
print(f"✅ Model tags configured")
print(f"   - best_training_run_id: {best_run_id}")
print(f"   - comparison_run_id: {comparison_run.info.run_id}")
print(f"   - experiment_group: {experiment_group_id}")

# Save best model locally
print(f"\n{'='*80}")
print("💾 SAVING MODEL LOCALLY")
print(f"{'='*80}")

best_model.save(
    "./best_gepa_rag_dir/",
    save_program=True
)
print("✅ Model saved to ./best_gepa_rag_dir/")

# ==================================================
# FINAL RESULTS SUMMARY
# ==================================================
print(f"\n{'='*80}")
print("🎉 TRAINING COMPLETE")
print(f"{'='*80}")
print(f"\n📊 Results Summary:")
print(f"   Baseline (Small LM):  {uncompiled_small_lm_accuracy:.2%}")
print(f"   Baseline (Large LM):  {uncompiled_large_lm_accuracy:.2%}")
print(f"   GEPA Custom AI Judge: {gepa_custom_accuracy:.2%}")
print(f"   GEPA SemanticF1:      {gepa_semantic_accuracy:.2%}")
print(f"\n🏆 Winner: {winner} ({best_accuracy:.2%})")
print(f"   Registered to UC: {FULL_MODEL_NAME} (v{model_version.version})")
print(f"   Alias: challenger")
print(f"\n🔗 Traceability:")
print(f"   Best Training Run: {best_run_id}")
print(f"   Comparison Run: {comparison_run.info.run_id}")
print(f"   Experiment Group: {experiment_group_id}")
print(f"{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display Final Results

# COMMAND ----------

# Generate table rows dynamically from model results (scalable!)
table_rows = "\n".join([
    f"""
        <tr style="background-color: {'#e8f5e9' if r['name'] == winner else 'white'};">
            <td style="padding: 10px; border: 1px solid #90caf9;">GEPA</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{r['name']}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{r['accuracy']:.2%}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{'🏆 Winner' if r['name'] == winner else ''}</td>
        </tr>
    """
    for r in sorted_results
])

# Display results in HTML
displayHTML(f"""
<div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f0f8ff; border-radius: 10px; border: 2px solid #4CAF50;">
    <h1 style="color: #2e7d32;">✅ Model Training & Registration Complete</h1>
    <hr style="border: 1px solid #4CAF50;">
    
    <h2 style="color: #1565c0;">🏆 Metric Comparison Results</h2>
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <tr style="background-color: #e3f2fd;">
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Training Method</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Metric Used</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Accuracy</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Selected</th>
        </tr>
        {table_rows}
    </table>
    
    <h2 style="color: #1565c0;">📦 Model Registration Details</h2>
    <ul style="font-size: 16px; line-height: 1.8;">
        <li><strong>Model URI:</strong> <code>models:/{FULL_MODEL_NAME}@challenger</code></li>
        <li><strong>Version:</strong> {model_version.version}</li>
        <li><strong>Optimization Method:</strong> GEPA</li>
        <li><strong>Winning Metric:</strong> {winner} ({best_accuracy:.2%})</li>
        <li><strong>Accuracy Improvement:</strong> {(best_accuracy - runner_up_accuracy):.2%} over runner-up</li>
        <li><strong>Deployment Alias:</strong> challenger</li>
        <li><strong>Status:</strong> Ready for testing</li>
    </ul>
    
    <h2 style="color: #1565c0;">🔗 Traceability</h2>
    <ul style="font-size: 16px; line-height: 1.8;">
        <li><strong>Best Training Run ID:</strong> <code>{best_run_id}</code></li>
        <li><strong>Comparison Run ID:</strong> <code>{comparison_run.info.run_id}</code></li>
        <li><strong>Experiment Group ID:</strong> <code>{experiment_group_id}</code></li>
        <li><strong>All Training Runs:</strong> {len(model_results)} models compared</li>
    </ul>
    
    <div style="margin-top: 20px; padding: 15px; background-color: #e8f5e9; border-left: 4px solid #4CAF50;">
        <strong>💡 Tip:</strong> Navigate to the model in Unity Catalog and check the <code>best_training_run_id</code> tag to view the training run that produced this model.
    </div>
</div>
""")