# main.py

from __future__ import annotations

import os
from pathlib import Path
from models import llama70b
from render.util import AttentionRenderer, RenderConfig


def main():
    # 1) Build/run the model (only HF token may come from env)
    llama = llama70b()
    llama.login_hf()                  # optional; uses HUGGINGFACE_TOKEN if present
    llama.config(key_scope="prompt")  # IMPORTANT for prompt-aligned attention
    llama.build()

    # 2) Build our image renderer
    renderer = AttentionRenderer(tokenizer=llama.tokenizer, config=RenderConfig(pool="all_layers_mean"))
    # code_set = []
    # for snippet in os.listdir("./Source"):
    #     code_set.append(open(f"./Source/{snippet}", 'r').read())
    # for code in code_set():

    output_root = Path('attn_viz')
    output_root.mkdir(parents=True, exist_ok=True)

    source_dir = Path("Source")
    snippet_paths = sorted(p for p in source_dir.iterdir() if p.is_file())
    if not snippet_paths:
        print(f"No code snippets found under {source_dir.resolve()}")
        llama.free()
        return
    instruction = "Summarize what this Java function does. Be concise and accurate."
    runs_per_snippet = 20
    for snippet_path in snippet_paths:
        code = snippet_path.read_text(encoding="utf-8")
        snippet_name = snippet_path.stem
        snippet_root = output_root / snippet_name
        snippet_root.mkdir(parents=True, exist_ok=True)
        for run_idx in range(1, runs_per_snippet + 1):
            result = llama.summarize_code(code)
            attn_map = renderer.map_attention_to_source(
                code_snippet=code,
                generation_result=result,
                instruction=instruction,
                pool="all_layers_mean",
            )
            run_dir = snippet_root / f"{run_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            renderer.save_text_dump(attn_map, str(run_dir / "attention.json"), str(run_dir / "attention.csv"))
            renderer.render_html(code, attn_map, str(run_dir / "attention.html"))
            renderer.render_png(code, attn_map, str(run_dir / "attention.png"))
            llama.save_dump(result, str(run_dir / "model_output.json"))
            print(f"[{snippet_name}] saved artifacts for run {run_idx:03d} -> {run_dir}")
    llama.free()

if __name__ == "__main__":
    main()
