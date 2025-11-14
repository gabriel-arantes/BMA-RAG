# Databricks notebook source
# MAGIC %pip install --index-url https://pypi.org/simple mlflow[databricks]>=3.1.0 \
# MAGIC     databricks-agents databricks-sdk dspy databricks-vectorsearch databricks-dspy pyyaml pydantic==2.12.4
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow

model_uri = f"models:/test_catalog.test_schema.bma_dspy_model@challenger"

# Download reqs
req_path = mlflow.pyfunc.get_model_dependencies(model_uri, format="pip")

# Install dependencies
os.system(f"pip install -r {req_path}")

dbutils.library.restartPython()

# COMMAND ----------

import sys
print(f"Python Version: {sys.version}")

# COMMAND ----------

import os
import mlflow
from mlflow.tracking import MlflowClient
from databricks import agents
from databricks.agents.evals import judges
import time
import requests
from typing import List, Dict
import uuid

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog_name", "test_catalog")
dbutils.widgets.text("schema_name", "test_schema")
dbutils.widgets.text("model_name", "bma_dspy_model")
dbutils.widgets.text("endpoint_name", "bma_dspy_endpoint")
dbutils.widgets.dropdown("scale_to_zero", "true", ["true", "false"])

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
model = dbutils.widgets.get("model_name")

FULL_MODEL_NAME = f"{catalog}.{schema}.{model}"
endpoint_name = dbutils.widgets.get("endpoint_name")
scale_to_zero = dbutils.widgets.get("scale_to_zero") == "true"

workspace_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
workspace_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print("Model:", FULL_MODEL_NAME)
print("Endpoint:", endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Dataset

# COMMAND ----------

eval_data = [
    {"q": "Which period did the AML/ATF audit cover?", "exp": "The audit covered April 1, 2021 to October 31, 2022."},
    {"q": "What was the overall audit rating for CTCBL?", "exp": "Satisfactory."},
    {"q": "Who is responsible for compliance policy amendments?", "exp": "Elkeisha Caisey, Senior Compliance Specialist."},
    {"q": "What is one area where internal controls need improvement?", "exp": "Internal controls and processes need to be performed in a consistent and timely manner."}
]

print("Evaluation examples:", len(eval_data))

# COMMAND ----------

# MAGIC %md
# MAGIC ## DSPy Loader and Inference

# COMMAND ----------

def load_dspy_alias(alias):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(FULL_MODEL_NAME, alias)
    uri = f"models:/{FULL_MODEL_NAME}/{mv.version}"
    print("Loading:", uri)

    model = mlflow.dspy.load_model(uri)
    return model, mv.version


def infer(model, q):
    inp = {"messages": [{"role": "user", "content": q}]}
    try:
        out = model.predict(inp)
    except:
        out = model(inp)

    if hasattr(out, "response"):
        return str(out.response)
    if isinstance(out, dict) and "response" in out:
        return str(out["response"])
    return str(out)


def evaluate(model):
    scores = []
    for item in eval_data:
        pred = infer(model, item["q"])
        j = judges.correctness(
            request=item["q"],
            response=pred,
            expected_response=item["exp"]
        )
        scores.append(1 if j.value == "yes" else 0)
    return sum(scores) / len(scores)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Champion vs Challenger

# COMMAND ----------

print("=== LOADING MODELS ===")

champion_model, champion_version = load_dspy_alias("champion")
challenger_model, challenger_version = load_dspy_alias("challenger")

print("Champion v", champion_version)
print("Challenger v", challenger_version)

print("\n=== Evaluating Champion ===")
acc_c = evaluate(champion_model)
print("Champion Accuracy:", acc_c)

print("\n=== Evaluating Challenger ===")
acc_h = evaluate(challenger_model)
print("Challenger Accuracy:", acc_h)

winner = "challenger" if acc_h > acc_c else "champion"
print("\nWinner:", winner)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote Challenger if wins

# COMMAND ----------

client = MlflowClient()

if winner == "challenger":
    print("Promoting challenger → champion")
    client.set_registered_model_alias(
        name=FULL_MODEL_NAME,
        alias="champion",
        version=challenger_version
    )
    champion_version = challenger_version
else:
    print("Champion retains title")

print("Champion now:", champion_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy using Agent Framework

# COMMAND ----------

deployment = agents.deploy(
    model_name=FULL_MODEL_NAME,        
    model_version=champion_version,
    scale_to_zero_enabled=scale_to_zero,
    service_principal_name="bma-dspy-serving"
)

print("Endpoint:", deployment.query_endpoint)