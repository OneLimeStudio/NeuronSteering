# ===== Google Colab: copy-paste this WHOLE file into ONE cell and press Run =====
# 1. In Colab: Runtime > Change runtime type > Hardware accelerator = T4 GPU
# 2. Paste this file, Run. Edit PROMPT below to change input.
# Yandere: L16N8927+0.3 L18N3719+0.5 L19N4087+0.8 L20N1950+0.4 L21N2465-0.4 L22N2354+0.15 L22N6408-0.4
# dose units: act[n] += dose * median(||resid leaving layer L||) / ||W_down[:, n]||, at the input of mlp.down_proj, every token

# --- install deps (Colab only, safe to re-run) ---
!pip install -q -U transformers accelerate

import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-4B"

# EDIT THIS:
PROMPT = "(prompt not shown in the source tweet)"

NEURONS = [  # (layer, neuron, dose)
    (16, 8927, 0.3),
    (18, 3719, 0.5),
    (19, 4087, 0.8),
    (20, 1950, 0.4),
    (21, 2465, -0.4),
    (22, 2354, 0.15),
    (22, 6408, -0.4),
]
# median ||residual leaving layer L|| from the published table (neuronary.org/norms/Qwen__Qwen3.5-4B.json)
MEDIAN = {16: 10.9506, 18: 13.8992, 19: 16.2707, 20: 18.0496, 21: 20.4401, 22: 24.8439}

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f" | {torch.cuda.get_device_name(0)}")
else:
    print(" | NO GPU - enable T4 GPU or it will be very slow / OOM")

# T4 (free Colab) has no bf16 -> use fp16. A100/L4 can use bf16.
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if use_bf16 else torch.float16
print(f"Using dtype: {DTYPE}")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=DTYPE, device_map="auto", trust_remote_code=True
    ).eval()
except Exception as e:
    print(f"ImageTextToText load failed ({e}), trying CausalLM...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, device_map="auto", trust_remote_code=True
    ).eval()

# Handle both Qwen2-VL-style (language_model.layers) and pure text (model.layers) layouts
if hasattr(model, "model") and hasattr(model.model, "language_model"):
    layers = model.model.language_model.layers
else:
    layers = model.model.layers

handles = []
for l, n, dose in NEURONS:
    w_norm = layers[l].mlp.down_proj.weight[:, n].float().norm().item()
    push = dose * MEDIAN[l] / w_norm
    print(f"L{l}N{n} dose={dose:+g} push={push:+.4f}")

    def hook(mod, inp, n=n, push=push):
        h = inp[0].clone()
        h[..., n] += push
        return (h,)

    handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(hook))

msgs = [{"role": "user", "content": PROMPT}]
enc = tok.apply_chat_template(
    msgs, add_generation_prompt=True, enable_thinking=False,
    return_tensors="pt", return_dict=True
).to(model.device)

with torch.no_grad():
    out = model.generate(**enc, do_sample=False, max_new_tokens=512)

print("\n===== OUTPUT =====")
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
