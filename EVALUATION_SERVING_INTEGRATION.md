# Champion vs Challenger Evaluation with Serving Integration

## Overview

The evaluation system has been enhanced to automatically deploy or update the champion model to a serving endpoint after the champion/challenger comparison. This ensures the winning model is immediately available for inference.

## Architecture

### Flow Diagram
```
1. Load Champion & Challenger models (by alias from Unity Catalog)
2. Evaluate both models using AI Judge (correctness metric)
3. Compare results
4. Promote winner to champion alias (if challenger wins)
5. Deploy/Update champion to serving endpoint
6. Test endpoint to ensure it's working
7. Generate evaluation report
```

## Components

### 1. Helper Module (`src/evaluation/helper.py`)
Contains minimal utilities needed for serving endpoint deployment:

- **`Tag` class**: Simple tagging for endpoint metadata
- **`deploy_model_serving_endpoint()`**: Creates or updates a serving endpoint
  - Detects if endpoint exists
  - Creates new endpoint if needed
  - Updates existing endpoint with new model version
- **`get_model_optimization_info()`**: Checks if model supports provisioned throughput
- **`test_serving_endpoint()`**: Validates endpoint is working with sample query

### 2. Evaluation Notebook (`src/evaluation/champion_challenger_evaluation.py`)
Enhanced with serving deployment capabilities:

**New Parameters:**
- `catalog_name`: Unity Catalog catalog (e.g., "test_catalog")
- `schema_name`: Unity Catalog schema (e.g., "test_schema")
- `model_name`: Model name (e.g., "bma_dspy_model")
- `serving_endpoint_name`: Name of the serving endpoint
- `serving_endpoint_workload_type`: GPU_SMALL, GPU_MEDIUM, GPU_LARGE, CPU
- `serving_endpoint_workload_size`: Small, Medium, Large
- `serving_endpoint_scale_to_zero`: Enable/disable scale-to-zero
- `inference_table_catalog/schema/name`: Optional inference table settings

**Note:** Workspace URL and authentication token are automatically obtained from the notebook context.

**Deployment Logic:**
1. After promotion, captures the promoted model version
2. Checks model optimization capabilities (provisioned throughput vs GPU)
3. Configures endpoint based on model type:
   - Optimizable models: Use provisioned throughput with calculated min/max
   - Non-optimizable: Standard GPU configuration with environment variables
4. Optionally enables inference table for monitoring
5. Creates or updates the endpoint
6. Waits for deployment to complete (15 seconds)
7. Tests endpoint with sample query
8. Logs deployment status to MLflow

**Error Handling:**
- Deployment wrapped in try/catch
- Evaluation completes successfully even if deployment fails
- Error details logged to MLflow for debugging

### 3. Workflow Configuration (`resources/champion_challenger_workflow.yml`)
Updated to pass endpoint configuration to evaluation task:

**Parameters Passed:**
- Model configuration (catalog, schema, model name)
- Endpoint configuration (all endpoint settings)
- Inference table configuration (optional)

**Variable References:**
Uses `${var.variable_name}` syntax to reference bundle variables from `databricks.yml`

### 4. Bundle Configuration (`databricks.yml`)
New variables added for endpoint configuration:

**Global Variables (with defaults):**
- `serving_endpoint_name`: "bma_champion_endpoint"
- `serving_endpoint_workload_type`: "GPU_SMALL"
- `serving_endpoint_workload_size`: "Small"
- `serving_endpoint_scale_to_zero`: "false"
- `inference_table_catalog/schema/name`: "" (optional, disabled by default)

**Environment-Specific Overrides:**
- **dev**: `serving_endpoint_name: "bma_champion_endpoint_dev"`
- **staging**: `serving_endpoint_name: "bma_champion_endpoint_staging"`
- **prod**: 
  - `serving_endpoint_name: "bma_champion_endpoint_prod"`
  - Inference table enabled: `prod.model_tracking.champion_model_inference`

**Authentication:**
Workspace URL and authentication tokens are automatically obtained from the notebook context using `dbutils.notebook.entry_point.getDbutils().notebook().getContext()`. No secret configuration required.

## Key Features

### 1. Flavor-Agnostic Model Loading
- Simple flavor detection by reading MLmodel YAML
- Uses unified pyfunc interface for inference
- Works with any MLflow model type (DSPy, LangChain, sklearn, etc.)

### 2. AI Judge Evaluation
- Uses Databricks AI Judge for correctness assessment
- Compares model responses to expected answers
- Binary scoring (yes/no) converted to accuracy metric

### 3. Automatic Promotion
- Challenger promoted to champion if accuracy is higher
- Champion retains title if no improvement
- Promotion metadata tracked (tags, date)

### 4. Serving Deployment/Update
- Automatically deploys or updates endpoint with champion model
- Supports both new endpoint creation and existing endpoint updates
- Handles different model optimization types:
  - Provisioned throughput for optimizable models
  - Standard GPU configuration for others

### 5. Environment Isolation
- Separate endpoints per environment (dev/staging/prod)
- Environment-specific configurations via bundle variables
- Inference table enabled only in production

### 6. Monitoring & Logging
- All evaluation metrics logged to MLflow
- Deployment status and errors tracked
- Optional inference tables for production monitoring
- HTML report with comprehensive results

## Usage

### Running the Evaluation

#### Via Databricks Workflow:
```bash
# Deploy and run the workflow
databricks bundle deploy -t dev
databricks bundle run champion_challenger_evaluation -t dev
```

#### Manually in Databricks Notebook:
1. Ensure champion and challenger models exist with appropriate aliases
2. Configure widget parameters (or use defaults)
3. Run all cells in the notebook

### Prerequisites

1. **Models in Unity Catalog:**
   - At least one model version with "champion" alias
   - At least one model version with "challenger" alias
   - Format: `{catalog}.{schema}.{model_name}`

2. **Permissions:**
   - Read access to Unity Catalog models
   - Create/update serving endpoints permission
   - (Optional) Create inference tables permission

**Note:** No additional secret configuration is needed. The notebook automatically uses the running user's authentication context.

### Configuration

#### Development:
- Quick testing with minimal resources
- No inference table
- Endpoint: `bma_champion_endpoint_dev`

#### Staging:
- Production-like testing
- No inference table
- Endpoint: `bma_champion_endpoint_staging`

#### Production:
- Full monitoring enabled
- Inference table: `prod.model_tracking.champion_model_inference`
- Endpoint: `bma_champion_endpoint_prod`

## Integration Points

### With Existing DSPy Model Training:
1. Train DSPy model using `author_model.py`
2. Register model to Unity Catalog
3. Set "challenger" alias on new version
4. Run evaluation workflow
5. If challenger wins, it's automatically promoted and deployed

### With LLMOps Pipeline:
- Uses similar serving deployment patterns
- Compatible with standard MLflow model format
- Supports provisioned throughput optimization
- Inference table structure compatible with monitoring pipelines

## Troubleshooting

### Common Issues:

**1. "NO_CHAMPION" exit:**
- No champion model exists yet
- Solution: Manually set champion alias on a model version first

**2. "NO_CHALLENGER" exit:**
- No challenger model found
- Solution: Train and register a new model with challenger alias

**3. Endpoint deployment fails:**
- Check permissions for endpoint creation
- Verify user has permission to create/update serving endpoints
- Review endpoint configuration (workload type/size)
- Check error message in MLflow logs

**4. Endpoint test fails:**
- Endpoint may need more time to be ready (increase wait time)
- Check endpoint logs in Databricks UI
- Verify model supports expected input format

## Future Enhancements

Potential improvements:
1. Support for multiple evaluation metrics (ROUGE, BLEU, etc.)
2. A/B testing capabilities with traffic splitting
3. Rollback mechanism if deployed model fails
4. Integration with human evaluation feedback loop
5. Automated retraining triggers based on performance degradation

## Files Modified/Created

### Created:
- `src/evaluation/helper.py` - Serving utilities

### Modified:
- `src/evaluation/champion_challenger_evaluation.py` - Added serving deployment
- `resources/champion_challenger_workflow.yml` - Added endpoint parameters
- `databricks.yml` - Added endpoint variables and environment configs

### Documentation:
- `EVALUATION_SERVING_INTEGRATION.md` - This file

