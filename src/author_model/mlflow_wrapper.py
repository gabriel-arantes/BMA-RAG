import mlflow
import dspy
import pandas as pd
import json
import os

# Imports
from dspy_program import RAG, BMAChatAssistant, extract_history_from_messages

class DspyModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """
        Esta função é chamada quando o modelo é carregado (ex: mlflow.pyfunc.load_model).
        Ela reconstrói nosso objeto RAG.
        """
        
        # 1. Carregar a config que salvamos
        config_path = os.path.join(context.artifacts["config"], "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 2. Configurar o LM do DSPy (ESSENCIAL)
        llm = dspy.LM(f"databricks/{config['lm_name']}")
        dspy.settings.configure(lm=llm)

        # 3. Instanciar um objeto RAG *vazio*
        self.program = RAG(
            lm_name=config["lm_name"],
            index_path=config["index_path"],
            max_history_length=config["max_history_length"],
            enable_history=config["enable_history"],
            for_mosaic_agent=True
        )
        
        # 4. Carregar o *estado treinado* (prompts otimizados)
        model_load_path = context.artifacts["compiled_model"]
        self.program.load(model_load_path)
        
        print("Modelo RAG DSPy carregado e configurado com sucesso.")

    def predict(self, context, model_input):
        """
        Esta função é chamada para inferência.
        'model_input' será um Pandas DataFrame.
        """
        predictions = []
        
        # O MLflow converte o JSON de entrada {'messages': [...]}
        # para um DataFrame. Acessamos a coluna 'messages'.
        for _, row in model_input.iterrows():
            try:
                # 'row["messages"]' deve conter a lista de dicts
                request_dict = {"messages": row["messages"]}
                response_obj = self.program.forward(request_dict)
                
                # Extrai a resposta final
                predictions.append(response_obj.response)
                
            except Exception as e:
                predictions.append(f"Error processing row: {e}")

        return pd.Series(predictions)