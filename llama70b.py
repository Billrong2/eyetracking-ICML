#!/usr/bin/env python3
# summarize_with_attn.py
# Usage examples:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_NAME=codellama/CodeLlama-70b-Instruct-hf python summarize_with_attn.py
#   CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_NAME=codellama/CodeLlama-70b-Instruct-hf USE_4BIT=1 MAX_NEW_TOKENS=96 python summarize_with_attn.py
#
# Notes:
# - Keep attn_implementation="eager" so attention weights are exposed.
# - 70B BF16 typically needs large VRAM (e.g., 4x80GB). Use USE_4BIT=1 to experiment on smaller GPUs.
# - Attention tensors scale ~O(L^2); long prompts or big MAX_NEW_TOKENS will increase memory.

import os
import json
import pprint
from typing import Dict, Any, List
from huggingface_hub import login
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
login(token = os.getenv("HUGGINGFACE_TOKEN"))
############################################
# Config (env overrides)
############################################
# MODEL_NAME = os.environ.get("MODEL_NAME", "codellama/CodeLlama-70b-Instruct-hf")
MODEL_NAME = "codellama/CodeLlama-70b-Instruct-hf"
# MODEL_DIR = os.environ.get("MODEL_DIR", "/data/xxr230000/model_cache/codellama_70b")  # your custom storage path
CACHE_DIR = "/data/xxr230000/model_cache/codellama_70b"
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"      # set to "1" to enable 4-bit quantized loading
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))  # 0 = deterministic
TOP_P = float(os.environ.get("TOP_P", "1.0"))
TOP_K = int(os.environ.get("TOP_K", "7"))  # 0 disables top_k
TOP_ATTENDED_K = int(os.environ.get("TOP_ATTENDED_K", "10"))  # how many top attended source tokens to keep per layer per step
ATTN_DUMP_DIR = os.environ.get("ATTN_DUMP_DIR", "attn_dumps")
MEM_FRACTION = float(os.environ.get("MEM_FRACTION", "0.90"))  # fraction of VRAM per GPU to allow

# Example code to summarize (replace this in your pipeline)
CODE_SNIPPET = r"""
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        y = target - x
        if y in seen:
            return [seen[y], i]
        seen[x] = i
    return None
"""

INSTRUCTION = (
    "Summarize what this Python function does. Be concise and accurate.\n\n"
    f"```python\n{CODE_SNIPPET}\n```"
)

# If using an instruct/chat model, plain text is usually fine for CodeLlama-Instruct.
PROMPT = INSTRUCTION


############################################
# Helpers
############################################
def auto_max_memory_dict(fraction: float = 0.90) -> Dict[int, str]:
    """
    Build a max_memory dict for all visible GPUs using a fraction of total VRAM.
    Example return: {0: '71GiB', 1: '71GiB', 2: '71GiB', 3: '71GiB'}
    """
    if not torch.cuda.is_available():
        return {}
    n = torch.cuda.device_count()
    mm: Dict[int, str] = {}
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        allowed = int(props.total_memory * fraction)
        mm[i] = f"{allowed // (1024**3)}GiB"
    return mm


def build_model_and_tokenizer() -> (AutoModelForCausalLM, AutoTokenizer):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir = CACHE_DIR  )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common = dict(
        device_map="auto",                      # shard across all visible GPUs
        max_memory=auto_max_memory_dict(MEM_FRACTION) if torch.cuda.is_available() else None,
        attn_implementation="eager",            # critical to expose attention tensors
    )

    if USE_4BIT:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            **common,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            **common,
        )

    model.eval()
    return model, tokenizer


def generate_with_attn(model, tokenizer, prompt: str) -> Dict[str, Any]:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[-1]

    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=(TEMPERATURE > 0),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=None if TOP_K <= 0 else TOP_K,
    )

    with torch.no_grad():
        out = model.generate(
            **enc,
            generation_config=gen_cfg,
            return_dict_in_generate=True,
            output_attentions=True,   # <-- capture attention per generated step
            output_scores=False,
            use_cache=True,
        )

    sequences = out.sequences  # [batch, prompt_len + gen_len]
    full_text = tokenizer.decode(sequences[0], skip_special_tokens=True)

    total_len = sequences.shape[-1]
    num_generated = total_len - prompt_len
    ids_all = sequences[0].tolist()
    tokens_all = tokenizer.convert_ids_to_tokens(ids_all)

    # Build compact attention summaries per generated token
    attn_record: List[Dict[str, Any]] = []

    # out.attentions is a tuple with length == num_generated steps
    # out.attentions[step][layer] is [batch=1, n_heads, q_len(=1), k_len]
    for step_idx in range(num_generated):
        # absolute token index in the full sequence for this generated token
        abs_idx = prompt_len + step_idx
        gen_tok = tokens_all[abs_idx]

        step_layers = out.attentions[step_idx]
        layer_summaries: List[Dict[str, Any]] = []

        for layer_idx, layer_tensor in enumerate(step_layers):
            attn = layer_tensor[0]         # [n_heads, 1, k_len]
            attn = attn[:, 0, :]           # [n_heads, k_len]
            attn_mean = attn.mean(dim=0).to(torch.float32)  # [k_len]

            k_len = attn_mean.shape[-1]
            top_k = min(TOP_ATTENDED_K, k_len)
            top_vals, top_idx = torch.topk(attn_mean, k=top_k, largest=True, sorted=True)

            top_idx = top_idx.tolist()
            top_vals = [float(v) for v in top_vals.tolist()]
            top_tokens = [tokens_all[i] for i in top_idx]

            layer_summaries.append({
                "layer": layer_idx,
                "k_len": int(k_len),
                "top_indices": top_idx,
                "top_tokens": top_tokens,
                "top_values": top_vals,
            })

        attn_record.append({
            "generated_step": step_idx,
            "absolute_token_index": abs_idx,
            "token": gen_tok,
            "layers": layer_summaries,
        })

    return {
        "model": MODEL_NAME,
        "use_4bit": USE_4BIT,
        "prompt": prompt,
        "prompt_length_tokens": prompt_len,
        "num_generated_tokens": num_generated,
        "tokens_all": tokens_all,
        "generated_text": full_text,
        "attention_by_generated_token": attn_record,
    }


def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"USE_4BIT={USE_4BIT}  |  GPUs visible={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        print("Per-GPU max_memory cap:", auto_max_memory_dict(MEM_FRACTION))

    model, tokenizer = build_model_and_tokenizer()

    # (Optional) print device map so you can verify all 4 GPUs are used
    try:
        print("\n=== Device map (layer -> GPU) ===")
        pprint.pprint(model.hf_device_map)
    except Exception:
        pass

    result = generate_with_attn(model, tokenizer, PROMPT)

    # Save JSON
    os.makedirs(ATTN_DUMP_DIR, exist_ok=True)
    json_path = os.path.join(ATTN_DUMP_DIR, "attn_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print short report
    print("\n=== GENERATED TEXT ===")
    print(result["generated_text"])

    print("\n=== ATTENTION SUMMARY (first 2 generated tokens, first 3 layers) ===")
    for rec in result["attention_by_generated_token"][:2]:
        print(f"\nToken #{rec['absolute_token_index']} -> {rec['token']!r}")
        for ls in rec["layers"][:3]:
            tops = ", ".join(
                f"{tok}:{val:.3f}"
                for tok, val in zip(ls["top_tokens"], ls["top_values"])
            )
            print(f"  Layer {ls['layer']:>2}: {tops}")
    print(f"\nSaved full JSON to: {json_path}")


if __name__ == "__main__":
    main()
