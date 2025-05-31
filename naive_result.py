import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache, LlamaForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

from hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import recommended_inductor_config_setter

from quant_cfg import get_quant_config_slm
import torch.nn.functional as F

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.


def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        # Prefill
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids


def speculative_generate(model, assistant_model, input_ids, past_key_values, assistant_past_key_values, max_new_tokens, max_draft_len):
    prompt_len = input_ids.shape[1]
    # prefill for the target model
    outputs = model.prefill_forward(
        input_ids,
        past_key_values=past_key_values,
        logits_to_keep=1
    )
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits, dim=-1) # [1, 1]
    
    # prefill for the assistant model
    assistant_outputs = assistant_model.prefill_forward(
        input_ids,
        past_key_values=assistant_past_key_values,
        logits_to_keep=1
    )
    assistant_past_key_values = assistant_outputs.past_key_values
    
    while 1:
        # save the pos and input_ids for target model
        target_pos = input_ids.shape[1]
        if target_pos - prompt_len >= max_new_tokens:
            return input_ids
        
        target_input_ids = input_ids.clone()
        target_next_tokens = next_token.clone()
        # token by token decoding for draft model
        draft_tokens = []
        draft_prob = []
        
        for i in range(max_draft_len):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            assistant_outputs = assistant_model(
                next_token,
                past_key_values=assistant_past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            
            logits = assistant_outputs.logits.clone()
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            assistant_past_key_values = assistant_outputs.past_key_values
            draft_tokens.append(next_token)
            
            
        draft_tokens = torch.cat(draft_tokens, dim=-1)
        draft_prob = torch.cat(draft_prob, dim=-2) # [1,draft_tokens,vocab_size]
        
        # compute the accept_len
        accept_len = 0
        cache_position = torch.arange(target_pos, target_pos + max_draft_len + 1, device=input_ids.device, dtype=torch.long)
        
        output = model.prefill_forward(
            torch.cat([target_next_tokens, draft_tokens], dim=-1),
            past_key_values=past_key_values,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position
        )
        
        past_key_values = output.past_key_values
        output_tokens = torch.argmax(output.logits, dim=-1)
        
        for i in range(max_draft_len):
            if output_tokens[0, i] == draft_tokens[0, i]:
                accept_len += 1
            else:
                break

        if (accept_len == 0):
            input_ids = torch.cat([target_input_ids, target_next_tokens], dim=-1)
            next_token = output_tokens[:, 0].unsqueeze(0)
        else:
            input_ids = torch.cat([target_input_ids, target_next_tokens, draft_tokens[:, :accept_len]], dim=-1)
            next_token = output_tokens[:, accept_len].unsqueeze(0)
        
def find_candidate_pred_tokens(
    input_ids: torch.Tensor,
    output_cache: torch.Tensor | None,
    max_ngram_size: int = 3,
    num_pred_tokens: int = 10,
    exclude_last_n: int = 0  # <== 新增參數
) -> torch.Tensor:
    device = input_ids.device
    empty = torch.tensor([], dtype=torch.long, device=device)

    if output_cache is None or output_cache.size(1) < 1:
        return empty

    cache_len = output_cache.size(1)
    if cache_len <= exclude_last_n:
        return empty

    # 切掉尾端剛 append 的 token
    search_cache = output_cache[:, :-exclude_last_n] if exclude_last_n > 0 else output_cache

    for n in range(max_ngram_size, 0, -1):
        if input_ids.size(1) < n or search_cache.size(1) < n:
            continue

        query = input_ids[0, -n:].unsqueeze(0)
        windows = search_cache.unfold(dimension=1, size=n, step=1)
        matches = (windows == query).all(dim=2)
        match_idxs = matches.nonzero(as_tuple=True)[1]

        for idx in reversed(match_idxs):  # 從最靠近尾端的 match 開始取
            start = idx + n
            end = start + num_pred_tokens
            if start < search_cache.size(1):
                end = min(end, search_cache.size(1))
                draft = search_cache[0, start:end]
                if draft.numel() > 0:
                    return draft

    return empty


def pld_generate(model, input_ids, past_key_values,max_new_tokens):
    # define parameters for prompt lookup
    max_draft_len = 1
    max_ngram_size = 2
    total_accept = 0
    prompt_len = input_ids.shape[1]

    # prefill for the target model
    outputs = model.prefill_forward(
        input_ids,
        past_key_values=past_key_values,
        logits_to_keep=1
    )
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits, dim=-1) # [1, 1]
    
    while 1:
        # save the pos and input_ids for target model
        target_pos = input_ids.shape[1]
        if target_pos - prompt_len >= max_new_tokens:
            print(total_accept)
            return input_ids
        
        target_input_ids = input_ids.clone()
        target_next_tokens = next_token.clone()

        # prompt lookup for draft model
        draft_tokens = find_candidate_pred_tokens(torch.cat([input_ids, target_next_tokens], dim = -1), input_ids, max_ngram_size=max_ngram_size, num_pred_tokens=max_draft_len)
        
        accept_len = 0
        if draft_tokens.numel() == 0: # no draft tokens found, pad to the expected length
            draft_tokens = torch.zeros((1, max_draft_len), dtype=torch.long, device=input_ids.device)
        else:
            if draft_tokens.ndim == 1:
                draft_tokens = draft_tokens.unsqueeze(0)
            if draft_tokens.shape[1] < max_draft_len: # padding
                padding = torch.zeros((1, max_draft_len - draft_tokens.shape[1]), dtype=draft_tokens.dtype, device=draft_tokens.device)
                draft_tokens = torch.cat([draft_tokens, padding], dim=-1)
        
        # compute the accept_len
        cache_position = torch.arange(target_pos, target_pos + max_draft_len + 1, device=input_ids.device, dtype=torch.long)
        
        output = model(
            torch.cat([target_next_tokens, draft_tokens], dim=-1),
            past_key_values=past_key_values,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position
        )
        
        past_key_values = output.past_key_values
        output_tokens = torch.argmax(output.logits, dim=-1)
        
        for j in range(max_draft_len):
            if draft_tokens[0, j] in torch.topk(output.logits[0,j], 1).indices:
                accept_len += 1
            else:
                break
        
        total_accept += accept_len
        if (accept_len == 0 or draft_tokens.numel() == 0):
            input_ids = torch.cat([target_input_ids, target_next_tokens], dim=-1)
            next_token = output_tokens[:, 0].unsqueeze(0)
        else:
            input_ids = torch.cat([target_input_ids, target_next_tokens, draft_tokens[:, :accept_len]], dim=-1)
            next_token = output_tokens[:, accept_len].unsqueeze(0)
        
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
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    
    ### === TODO: Load your model (you may change this part) ===
    # recommended_inductor_config_setter()
    backend = 'gemlite'
    
    model_name = "Tony027/Llama-3.2-3B-pruned-0.55-LoRA"
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    )
    
    quant_config = get_quant_config_slm(model)
    
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    from hqq.utils.patching import prepare_for_inference
    prepare_for_inference(model, backend=backend) 
    
    #####################################
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    model.prefill_forward = model.forward
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)
    

    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # === (Optional) Set up StaticCache for manual KV cache management ===
    max_cache_len = 32 + max_new_tokens
    past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len, device=model.device, dtype=model.dtype)
    ####################################################################
    
    for i in tqdm(range(5), desc="Warm Up..."):
        # === (Optional) Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        # generated = speculative_generate(model, model, input_ids, past_key_values, past_key_values, max_new_tokens, max_draft_len=1)
        # generated = pld_generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()
        
        
    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    
    for i in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Optional: Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        # generated = speculative_generate(model, model, input_ids, past_key_values, past_key_values, max_new_tokens, max_draft_len=1)
        # generated = pld_generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)
        
    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")
    
    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])
        
if __name__ == '__main__':
    main()
    
    
    
