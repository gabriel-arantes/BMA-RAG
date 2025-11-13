# Databricks notebook source
import mlflow
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel


# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

class Tag:
    """Simple tag class for endpoint tagging"""
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def as_dict(self):
        return {'key': self.key, 'value': self.value}


# COMMAND ----------

def deploy_model_serving_endpoint(endpoint_name, endpoint_config, host):
    """
    Deploy or update a model serving endpoint.
    
    Args:
        endpoint_name: Name of the serving endpoint
        endpoint_config: EndpointCoreConfigInput configuration
        host: Databricks workspace host URL
        
    Creates a new endpoint if it doesn't exist, otherwise updates the existing one.
    """
    # Initiate the workspace client
    w = WorkspaceClient()
    
    # Get endpoint if it exists
    existing_endpoint = next(
        (e for e in w.serving_endpoints.list() if e.name == endpoint_name), None
    )

    serving_endpoint_url = f"{host}/ml/endpoints/{endpoint_name}"

    # If endpoint doesn't exist, create it
    if existing_endpoint == None:

        # TODO: Tags can be parameterized
        tags = [Tag("team", "data science"), Tag("purpose", "champion_model")]
        print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
        w.serving_endpoints.create_and_wait(name=endpoint_name, config=endpoint_config, tags=tags)

        # TODO: Permissions should be parameterized
        print(f"Setting up permissions to the endpoint {endpoint_name}...")
        serving_endpoint_id = w.serving_endpoints.get(endpoint_name).id
        # Note: Update the user_name to appropriate users/groups for your environment
        # access_control_list=[
        #     ServingEndpointAccessControlRequest(
        #         user_name="user@example.com",
        #         permission_level=ServingEndpointPermissionLevel.CAN_VIEW
        #     )]
        # w.serving_endpoints.set_permissions(serving_endpoint_id=serving_endpoint_id, access_control_list=access_control_list)

    # If endpoint does exist, update it to serve the new version
    else:
        print(f"Updating the endpoint {serving_endpoint_url} to serve the new model version...")
        w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=endpoint_name)


# COMMAND ----------

def get_model_optimization_info(model_name, model_version, host, token):
    """
    Get model optimization information to determine if provisioned throughput is supported.
    
    Args:
        model_name: Full model name in Unity Catalog (catalog.schema.model)
        model_version: Model version number
        host: Databricks workspace host URL
        token: Authentication token
        
    Returns:
        dict: Optimization info including 'optimizable' flag and 'throughput_chunk_size'
    """
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}

    optimizable_info = requests.get(
        url=f"{host}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}", 
        headers=headers
    ).json()

    return optimizable_info


# COMMAND ----------

def test_serving_endpoint(endpoint_name, host, token):
    """
    Test the serving endpoint with a sample query to ensure it's working.
    
    Args:
        endpoint_name: Name of the serving endpoint
        host: Databricks workspace host URL
        token: Authentication token
        
    Raises:
        AssertionError: If the endpoint doesn't respond with status 200
    """
    data = {
        "messages": 
            [ 
             {
                 "role": "user", 
                 "content": "What is the audit rating?"
             }
            ]
           }
    
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}
    
    response = requests.post(
        url=f"{host}/serving-endpoints/{endpoint_name}/invocations", json=data, headers=headers
    )
    
    # Assert that the status code indicates success (2xx range)
    assert response.status_code == 200, f"Model Serving Endpoint: Expected status code 200 but got {response.status_code}"

    # You could also use the requests built-in success check:
    assert response.ok, f"Model Serving Endpoint: Request failed with status code {response.status_code}"
    
    print(f"✅ Endpoint test successful! Response preview: {response.text[:200]}...")

