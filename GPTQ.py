from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_id", type=str, required=True)
args = argparser.parse_args()

model_path = f"../Model/{args.model_id}"
quant_path = f"../Quantized/{args.model_id}-gptq"
hf_path = f"Tony027/{args.model_id}-gptq"

tokenizer = AutoTokenizer.from_pretrained(model_path)
gptq_config = GPTQConfig(bits=4, tokenizer=tokenizer, dataset="wikitext2",desc_act=False,)

quant_model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config = gptq_config, device_map = 'auto')

quant_model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)

api = HfApi()
try:
    api.repo_info(repo_id=hf_path, repo_type="model")
    print(f"Repository '{hf_path}' already exists. Using the existing repo.")
except HfHubHTTPError as e:
    if e.response.status_code == 404:
        api.create_repo(repo_id=hf_path, repo_type="model")
        print(f"Repository '{hf_path}' did not exist. Created a new repo.")

api.upload_folder(
    folder_path=quant_path,
    repo_id=hf_path,
    repo_type="model",
)