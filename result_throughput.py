import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
from vllm import LLM, SamplingParams

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

def generate_with_vllm(model, tokenizer, prompt, max_new_tokens):
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.7,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    outputs = model.generate(prompt, sampling_params)
    return outputs

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()


def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256
    device = 'cuda:0'
    
    model_name = "JCH25/Llama-3.2-3B-pruned-0.55-LoRA-gptqv2" # "Tony027/Llama-3.2-3b-Instruct-gptq" # "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Throughput Calculation with vLLM ---
    vllm_model = LLM(
        model=model_name,
        max_model_len=4096,
        quantization='gptq',
        max_num_seqs=1,
        swap_space=0.5,
        # enable_prefix_caching=True, #
        # speculative_config={
        #     "method" : "ngram",
        #     "num_speculative_tokens": 8,
        #     "prompt_lookup_max": 4,
        # },
        compilation_config={
            "level": 3,
            "use_cudagraph": True,
            "full_cuda_graph": True,
            "cudagraph_capture_sizes": [1], 
            "compile_sizes":[1],
            "cudagraph_num_of_warmups": 5,
        },
        disable_log_stats=True,
        dtype="float16",
        )
    
    warmup_prompt = "Explain what AI is."
    print("Warming up vLLM model...")
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    for _ in tqdm(range(5), desc="Warm Up vLLM..."): # Corrected tqdm range
        _ = generate_with_vllm(vllm_model, tokenizer, warmup_prompt, max_new_tokens)
        
    prompt_text = "How to learn a new language?"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    tputs = []
    time_record = []

    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        response = generate_with_vllm(vllm_model, tokenizer, prompt_text, max_new_tokens)
        
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    
        tput = max_new_tokens / (elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)


    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'\nPrompt: {prompt_text}\nResponse: {response[0].outputs[0].text}\n')

    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')

    # --- PPL Calculation with Hugging Face Model ---

    # hf_model_for_ppl = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map=device,
    #     torch_dtype=torch.float16,
    #     attn_implementation="eager" 
    # )
    # hf_model_for_ppl.eval()

    # ppl = evaluate_ppl(hf_model_for_ppl, tokenizer, device) # Store PPL result
    # print(f"Perplexity (PPL): {ppl}")

    # # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    # ppl = round(ppl, 2)

    with open("result.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        # writer.writerow(["value"])
        # writer.writerow([ppl])
        writer.writerow([rounded_tput])

if __name__ == '__main__':
    main()