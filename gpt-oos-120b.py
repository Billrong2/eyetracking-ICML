# Ensure all downloads go into your custom cache dir
import os
CACHE_DIR = "/data/xxr230000/model_cache/gpt-oss-120b"
os.makedirs(CACHE_DIR, exist_ok=True)

# (Optional but helpful) also point env vars at the same place
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = os.path.dirname(CACHE_DIR)

from huggingface_hub import login
# TIP: safer to read from an env var (e.g., os.environ["HF_TOKEN"]) instead of hardcoding
login(token=os.getenv("HUGGINGFACE_TOKEN"))

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
)

MODEL_ID = "openai/gpt-oss-120b"

# ----------------------------
# 1) High-level: pipeline()
# ----------------------------
# Pass cache_dir so it DOESN'T fall back to ~/.cache
pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    cache_dir=CACHE_DIR,
)

# Note: text-generation pipelines take strings, not chat message dicts
resp = pipe("Who are you?", max_new_tokens=40)
print("\n[Pipeline output]")
print(resp[0]["generated_text"])

# ----------------------------
# 2) Low-level: direct load
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

# Verify resolved local paths (should live under CACHE_DIR)
print("\n[Resolved paths]")
print("tokenizer:", tokenizer.name_or_path)
print("model:    ", model.name_or_path)

# If the model supports chat templates, you can use them like this:
messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print("\n[Generate output]")
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
