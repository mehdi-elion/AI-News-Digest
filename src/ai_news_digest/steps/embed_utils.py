# OPTION 1: https://huggingface.co/BAAI/bge-small-en-v1.5?inference_api=true


# OPTION 2: public HF serverless inference endpoint
# --> https://ui.endpoints.huggingface.co/mehdi-elion/endpoints/serverless
# --> https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient.feature_extraction
from huggingface_hub import InferenceClient

client = InferenceClient(model='BAAI/bge-small-en-v1.5')
response = client.feature_extraction('Hello world!')
print(response)


# OPTION 3: dedicated HF serverless Inference endpoint
# --> https://huggingface.co/docs/inference-endpoints/guides/custom_container


