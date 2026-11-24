# ===== Google Colab CELL 2: copy-paste this into a NEW cell, edit PROMPT, Run =====
# Assumes CELL 1 (main.py) already ran, so `tok` + `model` (with steering hooks) are in memory.
# No reload, no reinstall -> instant output.

# EDIT THIS ONLY:
PROMPT = "Hello, how are you?"
MAX_TOKENS = 512

import torch

# Safety check - model must already be loaded from main.py cell
if "model" not in globals() or "tok" not in globals():
    raise RuntimeError("Run CELL 1 (main.py) first - `model` / `tok` not found in session.")

msgs = [{"role": "user", "content": PROMPT}]
enc = tok.apply_chat_template(
    msgs, add_generation_prompt=True, enable_thinking=False,
    return_tensors="pt", return_dict=True
).to(model.device)

with torch.no_grad():
    out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_TOKENS)

print("\n===== OUTPUT =====")
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
