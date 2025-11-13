# Databricks notebook source
# MAGIC %pip install --index-url https://pypi.org/simple mlflow[databricks]>=3.1.0 databricks-agents databricks-sdk pyyaml dspy databricks-vectorsearch databricks-dspy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Champion vs Challenger Model Evaluation
# MAGIC 
# MAGIC This notebook compares the current **Champion** and **Challenger** models, automatically promotes the challenger to champion if it performs better, and deploys the winning model to a serving endpoint.
# MAGIC
# MAGIC ## How It Works
# MAGIC 1. Load both DSPy models by alias (champion and challenger)
# MAGIC 2. Evaluate both on the same test dataset using AI Judge
# MAGIC 3. Promote challenger to champion if it wins
# MAGIC 4. Deploy/update the champion model to a serving endpoint using Agent Framework
# MAGIC 5. Test the endpoint to ensure it's working

# COMMAND ----------

import mlflow
import pandas as pd
from typing import List, Dict
from databricks.agents.evals import judges
from databricks import agents

# Set MLflow registry to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Get parameters from widgets
dbutils.widgets.text("catalog_name", "test_catalog", "Catalog Name")
dbutils.widgets.text("schema_name", "test_schema", "Schema Name")
dbutils.widgets.text("model_name", "bma_dspy_model", "Model Name")

# Serving endpoint parameters
dbutils.widgets.text("serving_endpoint_name", "bma_dspy_endpoint", "Serving Endpoint Name")
dbutils.widgets.text("serving_endpoint_workload_type", "GPU_SMALL", "Workload Type (GPU_SMALL/GPU_MEDIUM/GPU_LARGE/CPU)")
dbutils.widgets.text("serving_endpoint_workload_size", "Small", "Workload Size (Small/Medium/Large)")
dbutils.widgets.dropdown("serving_endpoint_scale_to_zero", "true", ["true", "false"], "Scale to Zero")

# Optional inference table parameters (leave empty to skip)
dbutils.widgets.text("inference_table_catalog", "", "Inference Table Catalog (optional)")
dbutils.widgets.text("inference_table_schema", "", "Inference Table Schema (optional)")
dbutils.widgets.text("inference_table_name", "", "Inference Table Name (optional)")

# Get widget values
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
model_name = dbutils.widgets.get("model_name")

# Endpoint configuration
serving_endpoint_name = dbutils.widgets.get("serving_endpoint_name")
serving_endpoint_workload_type = dbutils.widgets.get("serving_endpoint_workload_type")
serving_endpoint_workload_size = dbutils.widgets.get("serving_endpoint_workload_size")
serving_endpoint_scale_to_zero = dbutils.widgets.get("serving_endpoint_scale_to_zero").lower() == "true"

# Inference table configuration (optional)
inference_table_catalog = dbutils.widgets.get("inference_table_catalog")
inference_table_schema = dbutils.widgets.get("inference_table_schema")
inference_table_name = dbutils.widgets.get("inference_table_name")

# Get workspace context (host and token from notebook)
workspace_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
workspace_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Full model name for Unity Catalog
FULL_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

print(f"Evaluating model: {FULL_MODEL_NAME}")
print(f"Champion: models:/{FULL_MODEL_NAME}@champion")
print(f"Challenger: models:/{FULL_MODEL_NAME}@challenger")
print(f"Serving endpoint: {serving_endpoint_name} at {workspace_host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Dataset
# MAGIC 
# MAGIC Using the same test data from model training for consistent evaluation.

# COMMAND ----------

# Evaluation dataset (same as used in training)
eval_data = [
    {"question": "Which period did the AML/ATF audit cover?", "expected": "The audit covered April 1, 2021 to October 31, 2022."},
    {"question": "What was the overall audit rating for CTCBL?", "expected": "Satisfactory."},
    {"question": "List the seven main pillars of the AML/ATF Program.", "expected": "Risk-Based Approach, Customer Due Diligence, Ongoing monitoring, Suspicious Activity Reporting, Record-Keeping, Training, Outsourcing and Reliance."},
    {"question": "Name one major audit finding and its rating.", "expected": "\"Noncompliance with Policies and Procedures\" was rated as Medium."},
    {"question": "Who is responsible for compliance policy amendments?", "expected": "Elkeisha Caisey, Senior Compliance Specialist."},
    {"question": "What is one area where internal controls need improvement?", "expected": "Internal controls and processes need to be performed in a consistent and timely manner."},
    {"question": "What were the main findings for Entity 35251?", "expected": "The Trigger Event Form and CRA were not completed or on file; ViewPoint was not updated for the retiring Protector."},
    {"question": "Who remediated the findings for Entity 35251?", "expected": "Natasha Hernandez, Head of Trust Services, CTCBL."},
    {"question": "What issue was found with distribution documentation for Entity 41683?", "expected": "Signed distribution task lists and Memorandum for the 6th distribution were missing; one distribution amount was incorrectly stated."},
    {"question": "How must distribution requests be verified?", "expected": "Signature verification via callback, Distribution Authentication stamp, Exception to Policy Form if needed, and minimum 3 signatures with one from Group Compliance."}
]

print(f"Loaded {len(eval_data)} evaluation examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## DSPy Model Loading
# MAGIC 
# MAGIC Load and evaluate DSPy models from Unity Catalog

# COMMAND ----------

def load_dspy_model_by_alias(model_name: str, alias: str):
    """
    Load DSPy model by alias from Unity Catalog.
    
    Returns:
        tuple: (model, version)
    """
    # Get version info
    client = mlflow.tracking.MlflowClient(registry_uri="databricks-uc")
    model_info = client.get_model_version_by_alias(model_name, alias)
    version = model_info.version
    
    # Load model with versioned URI (more reliable)
    model_uri = f"models:/{model_name}/{version}"
    
    print(f"   Loading model from URI: {model_uri}")
    print(f"   Testing model loading and inference...")
    
    try:
        # Load the model
        model = mlflow.dspy.load_model(model_uri)
        print(f"   ✅ Model loaded successfully")
        
        # Test with a simple inference to ensure it works
        test_input = {'messages': [{'content': 'Test question', 'role': 'user'}]}
        print(f"   Testing inference with sample input...")
        
        # Try the same inference approach we'll use in evaluation
        try:
            if hasattr(model, 'predict'):
                test_result = model.predict(test_input)
            else:
                test_result = model(test_input)
            print(f"   ✅ Inference test successful!")
            print(f"   Response type: {type(test_result)}")
        except Exception as inference_error:
            print(f"   ⚠️  Inference test failed: {inference_error}")
            print(f"   Will attempt alternative loading methods...")
            import traceback
            traceback.print_exc()
        
        return model, version
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def infer_dspy(model, input_data):
    """
    Run inference on a DSPy model.
    
    Args:
        model: Loaded DSPy model
        input_data: Dict with 'messages' key
        
    Returns:
        Model prediction output
    """
    # Try predict() method first (MLflow wrapper), fallback to direct call
    try:
        if hasattr(model, 'predict'):
            return model.predict(input_data)
        else:
            return model(input_data)
    except Exception as e:
        # If wrapped, try accessing underlying model
        if hasattr(model, '_model'):
            return model._model(input_data)
        elif hasattr(model, 'model'):
            return model.model(input_data)
        else:
            raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Function
# MAGIC 
# MAGIC Uses Databricks AI Judge for consistent quality assessment.

# COMMAND ----------

def evaluate_model(model, eval_dataset: List[Dict]) -> float:
    """
    Evaluate a DSPy model on the test dataset using AI Judge.
    
    Args:
        model: Loaded DSPy model
        eval_dataset: List of evaluation examples
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    scores = []
    
    for example in eval_dataset:
        question = example["question"]
        expected = example["expected"]
        
        # DSPy input format
        input_data = {'messages': [{'content': question, 'role': 'user'}]}
        
        try:
            # Get model prediction
            prediction = infer_dspy(model, input_data)
            
            # Extract response text from prediction
            if hasattr(prediction, 'response'):
                response_text = prediction.response
            elif isinstance(prediction, dict) and 'response' in prediction:
                response_text = prediction['response']
            elif isinstance(prediction, str):
                response_text = prediction
            else:
                response_text = str(prediction)
            
            # Use AI Judge to score correctness
            judgement = judges.correctness(
                request=question,
                response=response_text,
                expected_response=expected
            )
            
            # Score: 1 if correct, 0 if incorrect
            score = 1 if judgement.value == "yes" else 0
            scores.append(score)
            
        except Exception as e:
            print(f"Error evaluating question '{question}': {e}")
            scores.append(0)
    
    # Calculate accuracy
    accuracy = sum(scores) / len(scores) if scores else 0.0
    return accuracy

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Champion vs Challenger Evaluation

# COMMAND ----------

print("=" * 80)
print("🏆 CHAMPION VS CHALLENGER EVALUATION")
print("=" * 80)

# Start MLflow run to log evaluation results
with mlflow.start_run(run_name="champion_challenger_evaluation") as run:
    
    # Load Champion model
    print("\n📥 Loading Champion model...")
    try:
        champion_model, champion_version = load_dspy_model_by_alias(FULL_MODEL_NAME, "champion")
        print(f"✅ Champion: Version {champion_version}")
        mlflow.log_param("champion_version", champion_version)
        mlflow.log_param("model_type", "dspy")
    except Exception as e:
        print(f"❌ Error loading Champion: {e}")
        print("⚠️  Note: If this is the first run, no champion exists yet. Set the challenger as champion manually first.")
        #dbutils.notebook.exit("NO_CHAMPION")
    
    # Load Challenger model
    print("\n📥 Loading Challenger model...")
    try:
        challenger_model, challenger_version = load_dspy_model_by_alias(FULL_MODEL_NAME, "challenger")
        print(f"✅ Challenger: Version {challenger_version}")
        mlflow.log_param("challenger_version", challenger_version)
    except Exception as e:
        print(f"❌ Error loading Challenger: {e}")
        print("⚠️  No challenger model found. Nothing to evaluate.")
        #dbutils.notebook.exit("NO_CHALLENGER")

# COMMAND ----------

    # Evaluate Champion
    print("\n⚖️  Evaluating Champion...")
    champion_accuracy = evaluate_model(champion_model, eval_data)
    print(f"   Champion Accuracy: {champion_accuracy:.2%}")
    mlflow.log_metric("champion_accuracy", champion_accuracy)
    
    # Evaluate Challenger
    print("\n⚖️  Evaluating Challenger...")
    challenger_accuracy = evaluate_model(challenger_model, eval_data)
    print(f"   Challenger Accuracy: {challenger_accuracy:.2%}")
    mlflow.log_metric("challenger_accuracy", challenger_accuracy)
    
    # Determine winner
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"Champion (v{champion_version}):   {champion_accuracy:.2%}")
    print(f"Challenger (v{challenger_version}): {challenger_accuracy:.2%}")
    print(f"Difference:         {(challenger_accuracy - champion_accuracy):.2%}")
    
    # Promotion logic
    promoted_version = None
    if challenger_accuracy > champion_accuracy:
        winner = "challenger"
        print(f"\n🎉 CHALLENGER WINS! Promoting to Champion...")
        
        # Promote challenger to champion
        client = mlflow.tracking.MlflowClient(registry_uri="databricks-uc")
        client.set_registered_model_alias(
            name=FULL_MODEL_NAME,
            alias="champion",
            version=challenger_version
        )
        
        # Set promotion tags
        client.set_model_version_tag(
            name=FULL_MODEL_NAME,
            version=challenger_version,
            key="promoted_to_champion",
            value="true"
        )
        
        client.set_model_version_tag(
            name=FULL_MODEL_NAME,
            version=challenger_version,
            key="promotion_date",
            value=mlflow.get_run(run.info.run_id).data.tags.get("mlflow.startTime", "")
        )
        
        print(f"✅ Version {challenger_version} promoted to Champion")
        print(f"   Previous champion (v{champion_version}) has been replaced")
        promoted_version = challenger_version
        
    else:
        winner = "champion"
        print(f"\n🏆 CHAMPION RETAINS TITLE")
        print(f"   Champion (v{champion_version}) remains the best model")
        print(f"   Challenger (v{challenger_version}) did not outperform")
        promoted_version = champion_version
    
    # Log winner
    mlflow.log_param("winner", winner)
    mlflow.log_param("evaluation_dataset_size", len(eval_data))
    
    # Deploy/Update Model Serving Endpoint using Databricks Agent Framework
    print("\n" + "=" * 80)
    print("🚀 DEPLOYING CHAMPION MODEL USING DATABRICKS AGENT FRAMEWORK")
    print("=" * 80)
    
    try:
        import requests
        
        # Use Databricks Agent Framework for deployment
        # This handles DSPy signatures correctly and provides additional features
        print(f"\n🔧 Deploying champion model v{promoted_version} to endpoint...")
        print(f"   Model: {FULL_MODEL_NAME}")
        print(f"   Version: {promoted_version}")
        print(f"   Scale to zero: {serving_endpoint_scale_to_zero}")
        
        # Deploy using Agent Framework
        deployment = agents.deploy(
            model_name=FULL_MODEL_NAME,
            model_version=promoted_version,
            scale_to_zero_enabled=(serving_endpoint_scale_to_zero == "true")
        )
        
        print(f"\n✅ Endpoint deployed successfully!")
        print(f"   Endpoint URL: {deployment.query_endpoint}")
        
        # Test the deployed endpoint with Agent Framework format
        print(f"\n🧪 Testing endpoint...")
        
        # Agent Framework uses direct message format
        test_query = {
            "messages": [
                {"role": "user", "content": "What is the audit rating?"}
            ]
        }
        
        response = requests.post(
            deployment.query_endpoint,
            json=test_query,
            headers={"Authorization": f"Bearer {workspace_token}"}
        )
        
        if response.status_code == 200:
            print(f"✅ Endpoint test successful!")
            print(f"   Response preview: {response.text[:200]}...")
        else:
            print(f"⚠️  Endpoint test returned {response.status_code}: {response.text[:200]}...")
        
        mlflow.log_param("endpoint_deployed", True)
        mlflow.log_param("endpoint_name", serving_endpoint_name)
        mlflow.log_param("endpoint_url", deployment.query_endpoint)
        mlflow.log_param("deployed_model_version", promoted_version)
        mlflow.log_param("deployment_method", "agent_framework")
        
    except Exception as e:
        print(f"\n❌ Error deploying endpoint: {e}")
        print("   Evaluation completed successfully, but endpoint deployment failed.")
        mlflow.log_param("endpoint_deployed", False)
        mlflow.log_param("endpoint_error", str(e))
    
    # Display results
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

# Create HTML report
html_report = f"""
<div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f0f8ff; border-radius: 10px; border: 2px solid #4CAF50;">
    <h1 style="color: #2e7d32;">Champion vs Challenger Evaluation Results</h1>
    <hr style="border: 1px solid #4CAF50;">
    
    <h2 style="color: #1565c0;">Model Comparison</h2>
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <tr style="background-color: #e3f2fd;">
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Model</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Version</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Accuracy</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Result</th>
        </tr>
        <tr style="background-color: {'#e8f5e9' if winner == 'champion' else 'white'};">
            <td style="padding: 10px; border: 1px solid #90caf9;">Champion</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{champion_version}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{champion_accuracy:.2%}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{'🏆 Winner' if winner == 'champion' else ''}</td>
        </tr>
        <tr style="background-color: {'#e8f5e9' if winner == 'challenger' else 'white'};">
            <td style="padding: 10px; border: 1px solid #90caf9;">Challenger</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{challenger_version}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{challenger_accuracy:.2%}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{'🏆 Winner - Promoted!' if winner == 'challenger' else ''}</td>
        </tr>
    </table>
    
    <h2 style="color: #1565c0;">Evaluation Details</h2>
    <ul style="font-size: 16px; line-height: 1.8;">
        <li><strong>Model:</strong> {FULL_MODEL_NAME}</li>
        <li><strong>Evaluation Dataset:</strong> {len(eval_data)} examples</li>
        <li><strong>Metric:</strong> Correctness (AI Judge)</li>
        <li><strong>Performance Difference:</strong> {(challenger_accuracy - champion_accuracy):.2%}</li>
        <li><strong>Serving Endpoint:</strong> {serving_endpoint_name}</li>
        <li><strong>Deployed Version:</strong> {promoted_version}</li>
    </ul>
    
    {'<div style="margin-top: 20px; padding: 15px; background-color: #e8f5e9; border-left: 4px solid #4CAF50;"><strong>✅ Action Taken:</strong> Challenger promoted to Champion (version ' + str(challenger_version) + ')</div>' if winner == 'challenger' else '<div style="margin-top: 20px; padding: 15px; background-color: #fff3e0; border-left: 4px solid #ff9800;"><strong>ℹ️ No Action:</strong> Champion retains title</div>'}
</div>
"""

displayHTML(html_report)
