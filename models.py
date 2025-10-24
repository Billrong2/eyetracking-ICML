#!/usr/bin/env python3
# llama70b_attn.py
# Class-based wrapper for CodeLlama-70B-Instruct (or compatible HF causal LMs)
# with attention capture and pooled-attention summaries.
#
# Notes:
# - Only HUGGINGFACE_TOKEN is read from the environment (optional).
# - All other settings are hard-coded defaults; change them via llama.config(...).
# - Keep attn_implementation="eager" so attention tensors are exposed.
# - 70B BF16 typically needs large VRAM (e.g., 4x80GB). Set use_4bit=True to experiment on smaller GPUs.
# - Attention tensors scale ~O(L^2); long prompts or large max_new_tokens increase memory.

from __future__ import annotations
import os
import json
import pprint
from typing import Dict, Any, List, Tuple, Optional

from huggingface_hub import login
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

# ----------------------------
# Hard-coded defaults (change via llama.config(...))
# ----------------------------
DEFAULT_MODEL_NAME = "codellama/CodeLlama-70b-Instruct-hf"
DEFAULT_CACHE_DIR  = "/data/xxr230000/model_cache/codellama_70b"
DEFAULT_USE_4BIT   = False
DEFAULT_MAX_NEW    = 128
DEFAULT_TEMP       = 0.7
DEFAULT_TOP_P      = 1.0
DEFAULT_TOP_K      = 7
DEFAULT_TOP_ATTN_K = 10
DEFAULT_ATTN_DIR   = "attn_dumps"
DEFAULT_MEM_FRAC   = 0.90
DEFAULT_KEY_SCOPE  = "prompt"   # "prompt" | "all"
DEFAULT_RENORMAL   = True

class llama70b:
    """
    Class wrapper for CodeLlama-70B-Instruct with attention capture.

    Usage:
        llama = llama70b()
        llama.login_hf()  # optional; reads HUGGINGFACE_TOKEN if set
        llama.config(
            model_name="codellama/CodeLlama-70b-Instruct-hf",
            cache_dir="/path/to/cache",
            use_4bit=True,
            max_new_tokens=128,
            temperature=0.7,
            top_p=1.0,
            top_k=7,
            top_attended_k=10,
            attn_dump_dir="attn_dumps",
            mem_fraction=0.90,
            key_scope="prompt",   # or "all"
            renormalize=True,
        )
        llama.build()
        result = llama.summarize_code("def two_sum(...): ...")
        llama.save_dump(result, "attn_dumps/attn_summary.json")
        llama.free()
    """



    # ----------------------------
    # Construction / Configuration
    # ----------------------------
    def __init__(self):

        # hard-coded defaults (no env reads here)
        self.model_name: str = DEFAULT_MODEL_NAME
        self.cache_dir:  str = DEFAULT_CACHE_DIR
        self.use_4bit:   bool = DEFAULT_USE_4BIT
        self.max_new_tokens: int = DEFAULT_MAX_NEW
        self.temperature: float  = DEFAULT_TEMP
        self.top_p: float        = DEFAULT_TOP_P
        self.top_k: int          = DEFAULT_TOP_K
        self.top_attended_k: int = DEFAULT_TOP_ATTN_K
        self.attn_dump_dir: str  = DEFAULT_ATTN_DIR
        self.mem_fraction: float = DEFAULT_MEM_FRAC
        self.key_scope: str      = DEFAULT_KEY_SCOPE
        self.renormalize: bool   = DEFAULT_RENORMAL

        # internals
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def config(
        self,
        *,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_4bit: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        top_attended_k: Optional[int] = None,
        attn_dump_dir: Optional[str] = None,
        mem_fraction: Optional[float] = None,
        key_scope: Optional[str] = None,           # "prompt" | "all"
        renormalize: Optional[bool] = None,
    ) -> "llama70b":
        """Programmatic configuration (no environment variables used)."""
        if model_name is not None:     self.model_name = model_name
        if cache_dir is not None:      self.cache_dir = cache_dir
        if use_4bit is not None:       self.use_4bit = use_4bit
        if max_new_tokens is not None: self.max_new_tokens = max_new_tokens
        if temperature is not None:    self.temperature = temperature
        if top_p is not None:          self.top_p = top_p
        if top_k is not None:          self.top_k = top_k
        if top_attended_k is not None: self.top_attended_k = top_attended_k
        if attn_dump_dir is not None:  self.attn_dump_dir = attn_dump_dir
        if mem_fraction is not None:   self.mem_fraction = mem_fraction
        if key_scope is not None:      self.key_scope = key_scope.lower()
        if renormalize is not None:    self.renormalize = renormalize
        return self

    # ----------------------------
    # Auth (only HUGGINGFACE_TOKEN allowed from env)
    # ----------------------------
    def login_hf(self, token: Optional[str] = None):
        """
        Login to Hugging Face Hub. If token not provided explicitly,
        falls back to the HUGGINGFACE_TOKEN environment variable.
        """
        tok = token or os.getenv("HUGGINGFACE_TOKEN")
        if tok:
            login(token=tok)

    # ----------------------------
    # Build / Load
    # ----------------------------
    @staticmethod
    def _auto_max_memory_dict(fraction: float = 0.90) -> Dict[int, str]:
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

    def build(self):
        """Load tokenizer + model with attention enabled configuration."""
        print(f"Loading model: {self.model_name}")
        print(f"USE_4BIT={self.use_4bit}  |  GPUs visible={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        if torch.cuda.is_available():
            print("Per-GPU max_memory cap:", self._auto_max_memory_dict(self.mem_fraction))

        tok = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        common = dict(
            device_map="auto",  # shard across visible GPUs
            max_memory=self._auto_max_memory_dict(self.mem_fraction) if torch.cuda.is_available() else None,
            attn_implementation="eager",  # critical to expose attention tensors
            cache_dir=self.cache_dir,
        )

        if self.use_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                **common,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                **common,
            )

        model.eval()
        self.model = model
        self.tokenizer = tok

        # Optional: print device map
        try:
            print("\n=== Device map (layer -> GPU) ===")
            pprint.pprint(model.hf_device_map)
        except Exception:
            pass

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _renorm_or_copy(vec: torch.Tensor, do: bool) -> torch.Tensor:
        if not do:
            return vec
        s = float(vec.sum().item())
        if s > 1e-12:
            return vec / s
        return vec

    @staticmethod
    def _pool_layer_vectors(layer_vecs: List[torch.Tensor], mode: str) -> torch.Tensor:
        """
        layer_vecs: list of 1D tensors of length k_len (one per layer), already averaged across heads.
        mode: 'first', 'last', 'last4', 'all'
        """
        if len(layer_vecs) == 0:
            raise ValueError("No layer vectors to pool.")
        if mode == "first":
            return layer_vecs[0]
        if mode == "last":
            return layer_vecs[-1]
        if mode == "last4":
            take = min(4, len(layer_vecs))
            return torch.stack(layer_vecs[-take:], dim=0).mean(dim=0)
        if mode == "all":
            return torch.stack(layer_vecs, dim=0).mean(dim=0)
        raise ValueError(f"Unknown pooling mode: {mode}")

    @staticmethod
    def _to_cpu_list_floats(t: torch.Tensor) -> List[float]:
        return [float(x) for x in t.detach().to(torch.float32).cpu().tolist()]

    @staticmethod
    def _to_cpu_list_ints(t: torch.Tensor) -> List[int]:
        return [int(x) for x in t.detach().to(torch.int64).cpu().tolist()]

    # ----------------------------
    # Core: generate with attention
    # ----------------------------
    def _generate_with_attn(self, prompt: str) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None, "Call .build() first."

        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        prompt_len = input_ids.shape[-1]

        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=None if self.top_k <= 0 else self.top_k,
            cache_dir=self.cache_dir,
        )

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                generation_config=gen_cfg,
                return_dict_in_generate=True,
                output_attentions=True,   # <-- capture attention per generated step
                output_scores=False,
                use_cache=True,
            )

        sequences = out.sequences  # [batch, prompt_len + gen_len]
        full_text = self.tokenizer.decode(sequences[0], skip_special_tokens=True)

        total_len = sequences.shape[-1]
        num_generated = total_len - prompt_len
        ids_all = sequences[0].tolist()
        tokens_all = self.tokenizer.convert_ids_to_tokens(ids_all)

        # ---- Per-step pooled attention accumulators over the prompt (for global means)
        pool_names = ["first_layer", "last_layer", "last4_layers_mean", "all_layers_mean"]
        pool_modes = {
            "first_layer": "first",
            "last_layer": "last",
            "last4_layers_mean": "last4",
            "all_layers_mean": "all",
        }
        # accumulate mean distribution over prompt tokens across steps
        device = self.model.device
        global_prompt_accum = {name: torch.zeros(prompt_len, dtype=torch.float32, device=device) for name in pool_names}

        # ---- Records
        attn_record: List[Dict[str, Any]] = []   # per-layer tops
        pooled_record: List[Dict[str, Any]] = [] # per-step per-pool top-K over chosen key scope

        # out.attentions is a tuple with length == num_generated steps
        # out.attentions[step][layer] is [batch=1, n_heads, q_len(=1), k_len]
        for step_idx in range(num_generated):
            abs_idx = prompt_len + step_idx
            gen_tok = tokens_all[abs_idx]

            step_layers = out.attentions[step_idx]
            layer_summaries: List[Dict[str, Any]] = []
            layer_vecs: List[torch.Tensor] = []

            # --- Build per-layer mean-over-head vectors
            for layer_idx, layer_tensor in enumerate(step_layers):
                # [n_heads, 1, k_len] -> [n_heads, k_len] -> mean over heads -> [k_len]
                attn = layer_tensor[0]            # [n_heads, 1, k_len]
                attn = attn[:, 0, :]              # [n_heads, k_len]
                attn_mean = attn.mean(dim=0).to(torch.float32)  # [k_len]
                layer_vecs.append(attn_mean)

                # Keep diagnostic top-K (over full k_len for transparency)
                k_len = attn_mean.shape[-1]
                top_k = min(self.top_attended_k, k_len)
                top_vals, top_idx = torch.topk(attn_mean, k=top_k, largest=True, sorted=True)
                layer_summaries.append({
                    "layer": layer_idx,
                    "k_len": int(k_len),
                    "top_indices": self._to_cpu_list_ints(top_idx),
                    "top_tokens": [tokens_all[i] for i in self._to_cpu_list_ints(top_idx)],
                    "top_values": self._to_cpu_list_floats(top_vals),
                })

            # --- Pooled attention per strategy
            per_step_pooled = {"generated_step": step_idx, "absolute_token_index": abs_idx, "token": gen_tok, "pools": {}}

            # For global accumulation, consider prompt-token slice when KEY_SCOPE="prompt"
            for name, mode in pool_modes.items():
                pooled_vec = self._pool_layer_vectors(layer_vecs, mode=mode)  # [k_len]

                if self.key_scope == "prompt":
                    pooled_slice = pooled_vec[:prompt_len]
                    pooled_slice = self._renorm_or_copy(pooled_slice, self.renormalize)
                    # global accumulation over prompt (mean across steps; renorm at the end)
                    global_prompt_accum[name] += pooled_slice
                    # top-K within prompt
                    top_k = min(self.top_attended_k, prompt_len)
                    vals, idx = torch.topk(pooled_slice, k=top_k, largest=True, sorted=True)
                    idx_list = self._to_cpu_list_ints(idx)
                    val_list = self._to_cpu_list_floats(vals)
                    per_step_pooled["pools"][name] = {
                        "key_scope": "prompt",
                        "top_indices": idx_list,                 # indices within prompt range [0, prompt_len)
                        "top_tokens": [tokens_all[i] for i in idx_list],
                        "top_values": val_list,
                    }
                else:
                    # KEY_SCOPE == "all"
                    pooled_vec = self._renorm_or_copy(pooled_vec, self.renormalize)
                    k_len = pooled_vec.shape[-1]
                    top_k = min(self.top_attended_k, k_len)
                    vals, idx = torch.topk(pooled_vec, k=top_k, largest=True, sorted=True)
                    idx_list = self._to_cpu_list_ints(idx)
                    val_list = self._to_cpu_list_floats(vals)
                    per_step_pooled["pools"][name] = {
                        "key_scope": "all",
                        "top_indices": idx_list,                 # absolute indices into tokens_all
                        "top_tokens": [tokens_all[i] for i in idx_list],
                        "top_values": val_list,
                    }

            pooled_record.append(per_step_pooled)

            # Keep per-layer top-K diagnostic
            attn_record.append({
                "generated_step": step_idx,
                "absolute_token_index": abs_idx,
                "token": gen_tok,
                "layers": layer_summaries,
            })

        # --- Build global pooled distributions over prompt tokens (mean across steps)
        global_pooled_over_prompt = {}
        if self.key_scope == "prompt" and num_generated > 0:
            prompt_tokens = tokens_all[:prompt_len]
            for name in pool_names:
                mean_vec = global_prompt_accum[name] / float(num_generated)  # average across steps
                mean_vec = self._renorm_or_copy(mean_vec, self.renormalize)
                # top-K for reporting
                top_k = min(self.top_attended_k, prompt_len)
                vals, idx = torch.topk(mean_vec, k=top_k, largest=True, sorted=True)
                idx_list = self._to_cpu_list_ints(idx)
                val_list = self._to_cpu_list_floats(vals)
                global_pooled_over_prompt[name] = {
                    "prompt_tokens": prompt_tokens,
                    "scores": self._to_cpu_list_floats(mean_vec),  # full length prompt_len
                    "top_indices": idx_list,
                    "top_tokens": [prompt_tokens[i] for i in idx_list],
                    "top_values": val_list,
                }

        return {
            "model": self.model_name,
            "use_4bit": self.use_4bit,
            "prompt_length_tokens": prompt_len,
            "num_generated_tokens": num_generated,
            "tokens_all": tokens_all,
            "generated_text": full_text,
            # per-layer diagnostic (kept)
            "attention_by_generated_token": attn_record,
            # per-step per-pool top-K
            "pooled_attention_by_generated_token": pooled_record,
            # global per-pool distribution over prompt tokens (only when KEY_SCOPE='prompt')
            "global_pooled_attention_over_prompt": global_pooled_over_prompt,
            "pooling_strategies": {
                "first_layer": "first",
                "last_layer": "last",
                "last4_layers_mean": "last4",
                "all_layers_mean": "all",
            },
            "key_scope": self.key_scope,
            "renormalized": self.renormalize,
        }

    # ----------------------------
    # Public interface
    # ----------------------------
    def summarize_code(self, code_snippet: str, instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Provide a code snippet; returns the attention dump structure.
        """
        if instruction is None:
            instruction = "Summarize what this Python function does. Be concise and accurate."

        prompt = (
            f"{instruction}\n\n"
            f"```python\n{code_snippet}\n```"
        )
        return self._generate_with_attn(prompt)

    def save_dump(self, result: Dict[str, Any], json_path: str) -> str:
        """Persist a result to JSON and return the path."""
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return json_path

    def free(self):
        """Release model and VRAM."""
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Optional alias
Llama70B = llama70b


# ----------------------------
# Main (demo)
# ----------------------------
def _short_preview(result: Dict[str, Any], n_steps: int = 2, n_layers: int = 3, n_top: int = 5):
    print("\n=== GENERATED TEXT ===")
    print(result["generated_text"])

    print("\n=== ATTENTION SUMMARY (first %d generated tokens, first %d layers) ===" % (n_steps, n_layers))
    for rec in result["attention_by_generated_token"][:n_steps]:
        idx = rec.get("absolute_token_index", -1)
        print(f"\nToken #{idx} -> {rec['token']!r}")
        for ls in rec["layers"][:n_layers]:
            tops = ", ".join(
                f"{tok}:{val:.3f}"
                for tok, val in zip(ls["top_tokens"], ls["top_values"])
            )
            print(f"  Layer {ls['layer']:>2}: {tops}")

    print("\n=== POOLED (first %d generated tokens; top-%d per strategy, scope=%s) ==="
          % (n_steps, n_top, result["key_scope"]))
    for rec in result["pooled_attention_by_generated_token"][:n_steps]:
        print(f"\nToken #{rec['absolute_token_index']} -> {rec['token']!r}")
        for name, payload in rec["pools"].items():
            tops = ", ".join(
                f"{tok}:{val:.3f}"
                for tok, val in zip(payload["top_tokens"][:n_top], payload["top_values"][:n_top])
            )
            print(f"  {name:>22}: {tops}")

    if result["key_scope"] == "prompt" and result["global_pooled_attention_over_prompt"]:
        print("\n=== GLOBAL POOLED OVER PROMPT (top-10) ===")
        for name, payload in result["global_pooled_attention_over_prompt"].items():
            tops = ", ".join(
                f"{tok}:{val:.3f}"
                for tok, val in zip(payload["top_tokens"][:10], payload["top_values"][:10])
            )
            print(f"  {name:>22}: {tops}")


def main():
    # Example usage (adjust config here; no env overrides except optional HUGGINGFACE_TOKEN for login)
    llama = llama70b()

    # Optional: login using HUGGINGFACE_TOKEN if present
    llama.login_hf()

    # Configure as needed (edit these lines, or call from your code)
    llama.config(
        model_name="/data/xxr230000/model_cache/codellama_70b/models--codellama--CodeLlama-70b-Instruct-hf/snapshots/397cae981dffaf5d5c9c90e89a0a75a850528b70",
        cache_dir=DEFAULT_CACHE_DIR,
        use_4bit=DEFAULT_USE_4BIT,
        max_new_tokens=DEFAULT_MAX_NEW,
        temperature=DEFAULT_TEMP,
        top_p=DEFAULT_TOP_P,
        top_k=DEFAULT_TOP_K,
        top_attended_k=DEFAULT_TOP_ATTN_K,
        attn_dump_dir=DEFAULT_ATTN_DIR,
        mem_fraction=DEFAULT_MEM_FRAC,
        key_scope=DEFAULT_KEY_SCOPE,
        renormalize=DEFAULT_RENORMAL,
    )

    print(f"Cache dir: {llama.cache_dir}")
    if torch.cuda.is_available():
        print("CUDA devices:", torch.cuda.device_count())

    llama.build()

    demo_code = r"""
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        y = target - x
        if y in seen:
            return [seen[y], i]
        seen[x] = i
    return None
"""

    result = llama.summarize_code(demo_code)

    # Short, human-readable preview
    _short_preview(result, n_steps=2, n_layers=3, n_top=5)

    # Save full JSON
    os.makedirs(llama.attn_dump_dir, exist_ok=True)
    out_path = os.path.join(llama.attn_dump_dir, "attn_summary.json")
    llama.save_dump(result, out_path)
    print(f"\nSaved full JSON to: {out_path}")

    # Free VRAM/resources if desired
    llama.free()


if __name__ == "__main__":
    main()
