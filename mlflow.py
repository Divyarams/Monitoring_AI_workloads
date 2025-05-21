import boto3
import mlflow
import time
import json

# Initialize clients
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
mlflow.set_tracking_uri("http://localhost:5000")  # or your MLflow server

def track_bedrock_invocation(prompt, model_id="meta.llama3-2-1b-instruct-v1:0", **kwargs):
    with mlflow.start_run():
        # Log input parameters
        mlflow.log_params({
            "model_id": model_id,
            **kwargs
        })
        mlflow.log_text(prompt, "prompt.txt")
        
        # Prepare request
        body = {
            "prompt": prompt,
            **{k: v for k, v in kwargs.items() if v is not None}
        }
        
        # Track invocation
        start_time = time.time()
        try:
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(body))
            
            result = json.loads(response['body'].read())
            completion = result['completion']
            
            # Log results
            mlflow.log_metrics({
                "latency_seconds": time.time() - start_time,
                "response_length": len(completion)
            })
            mlflow.log_text(completion, "response.txt")
            
            return completion
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            raise


if __name__ == "__main__":
    response = track_bedrock_invocation(
        prompt="Explain quantum computing to a 5-year-old",
        model_id="meta.llama3-2-1b-instruct-v1:0",
        max_tokens_to_sample=300,
        temperature=0.1
    )
    print(response)
