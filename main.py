# main.py

from __future__ import annotations

import os
from models import llama70b
# from attention_renderer import AttentionRenderer, RenderConfig
from render.util import AttentionRenderer, RenderConfig


def main():
    # 1) Build/run the model (only HF token may come from env)
    llama = llama70b()
    llama.login_hf()                  # optional; uses HUGGINGFACE_TOKEN if present
    llama.config(key_scope="prompt")  # IMPORTANT for prompt-aligned attention
    llama.build()

    code = """\
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        y = target - x
        if y in seen:
            return [seen[y], i]
        seen[x] = i
    return None
"""

    # 2) Generate with attention
    result = llama.summarize_code(code)

    # 3) Create renderer, *pass tokenizer* from llama
    renderer = AttentionRenderer(tokenizer=llama.tokenizer, config=RenderConfig(pool="all_layers_mean"))

    # 4) Map attention scores back to original source
    attn_map = renderer.map_attention_to_source(
        code_snippet=code,
        generation_result=result,
        instruction="Summarize what this Python function does. Be concise and accurate.",
        pool="all_layers_mean",
    )

    # 5) Persist numeric dump + visuals
    os.makedirs("attn_viz", exist_ok=True)
    renderer.save_text_dump(attn_map, "attn_viz/attention.json", "attn_viz/attention.csv")
    renderer.render_html(code, attn_map, "attn_viz/attention.html")
    renderer.render_png(code, attn_map, "attn_viz/attention.png")

    print("Saved: attn_viz/attention.{json,csv,html,png}")


if __name__ == "__main__":
    main()
