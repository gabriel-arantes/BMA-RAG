# Databricks notebook source
# MAGIC %pip install --index-url https://pypi.org/simple mlflow[databricks]>=3.1.0 databricks-agents databricks-sdk pyyaml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./helper

# COMMAND ----------

# MAGIC %md
# MAGIC # Champion vs Challenger Model Evaluation
# MAGIC 
# MAGIC This notebook compares the current **Champion** and **Challenger** models, automatically promotes the challenger to champion if it performs better, and deploys the winning model to a serving endpoint.
# MAGIC
# MAGIC ## How It Works
# MAGIC 1. Load both models by alias (champion and challenger)
# MAGIC 2. Auto-detect model flavor (works with any MLflow model)
# MAGIC 3. Evaluate both on the same test dataset using AI Judge
# MAGIC 4. Promote challenger to champion if it wins
# MAGIC 5. Deploy/update the champion model to a serving endpoint
# MAGIC 6. Test the endpoint to ensure it's working

# COMMAND ----------

import mlflow
import yaml
from typing import List, Dict
from databricks.agents.evals import judges

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
dbutils.widgets.text("serving_endpoint_name", "bma_champion_endpoint", "Serving Endpoint Name")
dbutils.widgets.text("serving_endpoint_workload_type", "GPU_SMALL", "Workload Type (GPU_SMALL/GPU_MEDIUM/GPU_LARGE/CPU)")
dbutils.widgets.text("serving_endpoint_workload_size", "Small", "Workload Size (Small/Medium/Large)")
dbutils.widgets.dropdown("serving_endpoint_scale_to_zero", "false", ["true", "false"], "Scale to Zero")

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
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
workspace_host = f"https://{workspace_url}"
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
# MAGIC ## Flavor-Agnostic Model Loading
# MAGIC 
# MAGIC These functions automatically detect and load any MLflow model type (pyfunc, sklearn, langchain, DSPy, etc.)

# COMMAND ----------

def load_model_info_for_alias(model_name: str, alias: str):
    """
    Load model by alias and auto-identify its flavor.
    
    Returns:
        tuple: (model_uri, flavor, version)
    """
    # Get version info first
    client = mlflow.tracking.MlflowClient(registry_uri="databricks-uc")
    model_info = client.get_model_version_by_alias(model_name, alias)
    version = model_info.version
    
    # Use versioned URI for artifact download (more reliable than alias)
    versioned_uri = f"models:/{model_name}/{version}"
    
    # Download MLmodel YAML to detect flavor
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=versioned_uri + "/MLmodel")
    
    with open(local_path, "r") as f:
        mlmodel_yaml = yaml.safe_load(f)
    
    # Get model flavor
    flavors = list(mlmodel_yaml.get("flavors", {}).keys())
    flavor = flavors[0] if flavors else "pyfunc"
    
    # Return with alias-based URI for consistency
    model_uri = f"models:/{model_name}@{alias}"
    
    return model_uri, flavor, version


def infer(model_uri: str, flavor: str, input_data):
    """
    Flavor-agnostic inference function using MLflow's pyfunc interface.
    
    All MLflow models (regardless of flavor) provide a unified pyfunc interface.
    This approach avoids serialization issues and works consistently across all model types.
    
    Args:
        model_uri: MLflow model URI
        flavor: Model flavor (for logging/display only, not used for loading)
        input_data: Input for prediction (format depends on model)
        
    Returns:
        Model prediction output
    """
    # Always use pyfunc interface - it's the standard MLflow inference API
    model = mlflow.pyfunc.load_model(model_uri)
    return model.predict(input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Function
# MAGIC 
# MAGIC Uses Databricks AI Judge for consistent quality assessment.

# COMMAND ----------

def evaluate_model(model_uri: str, flavor: str, eval_dataset: List[Dict]) -> float:
    """
    Evaluate a model on the test dataset using AI Judge.
    
    Args:
        model_uri: MLflow model URI
        flavor: Model flavor
        eval_dataset: List of evaluation examples
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    scores = []
    
    for example in eval_dataset:
        question = example["question"]
        expected = example["expected"]
        
        # Format input for DSPy models (supports Mosaic Agent format)
        input_data = {'messages': [{'content': question, 'role': 'user'}]}
        
        try:
            # Get model prediction
            prediction = infer(model_uri, flavor, input_data)
            
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
        champion_uri, champion_flavor, champion_version = load_model_info_for_alias(FULL_MODEL_NAME, "champion")
        print(f"✅ Champion: Version {champion_version} (Flavor: {champion_flavor})")
        mlflow.log_param("champion_version", champion_version)
        mlflow.log_param("champion_flavor", champion_flavor)
    except Exception as e:
        print(f"❌ Error loading Champion: {e}")
        print("⚠️  Note: If this is the first run, no champion exists yet. Set the challenger as champion manually first.")
        dbutils.notebook.exit("NO_CHAMPION")
    
    # Load Challenger model
    print("\n📥 Loading Challenger model...")
    try:
        challenger_uri, challenger_flavor, challenger_version = load_model_info_for_alias(FULL_MODEL_NAME, "challenger")
        print(f"✅ Challenger: Version {challenger_version} (Flavor: {challenger_flavor})")
        mlflow.log_param("challenger_version", challenger_version)
        mlflow.log_param("challenger_flavor", challenger_flavor)
    except Exception as e:
        print(f"❌ Error loading Challenger: {e}")
        print("⚠️  No challenger model found. Nothing to evaluate.")
        dbutils.notebook.exit("NO_CHALLENGER")
    
    # Evaluate Champion
    print("\n⚖️  Evaluating Champion...")
    champion_accuracy = evaluate_model(champion_uri, champion_flavor, eval_data)
    print(f"   Champion Accuracy: {champion_accuracy:.2%}")
    mlflow.log_metric("champion_accuracy", champion_accuracy)
    
    # Evaluate Challenger
    print("\n⚖️  Evaluating Challenger...")
    challenger_accuracy = evaluate_model(challenger_uri, challenger_flavor, eval_data)
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
    
    # Deploy/Update Model Serving Endpoint
    print("\n" + "=" * 80)
    print("🚀 DEPLOYING CHAMPION MODEL TO SERVING ENDPOINT")
    print("=" * 80)
    
    try:
        # Import serving utilities
        from databricks.sdk.service.serving import EndpointCoreConfigInput
        import time
        
        # Get model optimization info
        print(f"\n📊 Checking model optimization capabilities for version {promoted_version}...")
        optimizable_info = get_model_optimization_info(FULL_MODEL_NAME, promoted_version, workspace_host, workspace_token)
        is_optimizable = optimizable_info.get("optimizable", False)
        
        print(f"   Optimizable: {is_optimizable}")
        mlflow.log_param("model_optimizable", is_optimizable)
        
        # Build endpoint configuration
        endpoint_config_dict = {}
        
        if is_optimizable:
            # Use provisioned throughput for optimizable models
            chunk_size = optimizable_info['throughput_chunk_size']
            min_provisioned_throughput = 0
            max_provisioned_throughput = 2 * chunk_size
            
            print(f"   Using provisioned throughput: {min_provisioned_throughput} - {max_provisioned_throughput}")
            
            endpoint_config_dict = { 
                "served_entities": [
                    {
                        "entity_name": FULL_MODEL_NAME,
                        "entity_version": promoted_version,
                        "workload_size": serving_endpoint_workload_size,
                        "scale_to_zero_enabled": serving_endpoint_scale_to_zero,
                        "workload_type": serving_endpoint_workload_type,
                        "min_provisioned_throughput": min_provisioned_throughput,
                        "max_provisioned_throughput": max_provisioned_throughput
                    }
                ]
            }
        else:
            # Standard GPU configuration for non-optimizable models
            print(f"   Using standard GPU configuration: {serving_endpoint_workload_type} / {serving_endpoint_workload_size}")
            
            endpoint_config_dict = { 
                "served_entities": [
                    {
                        "entity_name": FULL_MODEL_NAME,
                        "entity_version": promoted_version,
                        "workload_size": serving_endpoint_workload_size,
                        "scale_to_zero_enabled": serving_endpoint_scale_to_zero,
                        "workload_type": serving_endpoint_workload_type,
                        "environment_vars": {
                            "DATABRICKS_HOST": f"{workspace_host}",
                            "DATABRICKS_TOKEN": f"{workspace_token}"
                        }
                    }
                ]
            }
        
        # Add inference table configuration if provided
        if inference_table_catalog and inference_table_schema and inference_table_name:
            print(f"   Enabling inference table: {inference_table_catalog}.{inference_table_schema}.{inference_table_name}")
            endpoint_config_dict["auto_capture_config"] = {
                "catalog_name": f"{inference_table_catalog}",
                "schema_name": f"{inference_table_schema}",
                "table_name_prefix": f"{inference_table_name}"
            }
            mlflow.log_param("inference_table_enabled", True)
        else:
            mlflow.log_param("inference_table_enabled", False)
        
        endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)
        
        # Deploy or update the endpoint
        print(f"\n🔧 Deploying champion model (v{promoted_version}) to endpoint: {serving_endpoint_name}")
        deploy_model_serving_endpoint(serving_endpoint_name, endpoint_config, workspace_host)
        
        # Wait a bit for the endpoint to be ready
        print("\n⏳ Waiting for endpoint to be ready...")
        time.sleep(15)
        
        # Test the endpoint
        print(f"\n🧪 Testing endpoint: {serving_endpoint_name}")
        test_serving_endpoint(serving_endpoint_name, workspace_host, workspace_token)
        
        print(f"\n✅ Endpoint {serving_endpoint_name} is live and serving champion model v{promoted_version}")
        mlflow.log_param("endpoint_deployed", True)
        mlflow.log_param("endpoint_name", serving_endpoint_name)
        mlflow.log_param("deployed_model_version", promoted_version)
        
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
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Flavor</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Accuracy</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #90caf9;">Result</th>
        </tr>
        <tr style="background-color: {'#e8f5e9' if winner == 'champion' else 'white'};">
            <td style="padding: 10px; border: 1px solid #90caf9;">Champion</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{champion_version}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{champion_flavor}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{champion_accuracy:.2%}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{'🏆 Winner' if winner == 'champion' else ''}</td>
        </tr>
        <tr style="background-color: {'#e8f5e9' if winner == 'challenger' else 'white'};">
            <td style="padding: 10px; border: 1px solid #90caf9;">Challenger</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{challenger_version}</td>
            <td style="padding: 10px; border: 1px solid #90caf9;">{challenger_flavor}</td>
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
