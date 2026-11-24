# ===== Google Colab CELL: 1 prompt -> answers from ALL profiles =====
# Paste this into ONE new cell. Assumes main.py cell already ran (reuses `model`/`tok`).
# If you restarted runtime, it will auto-load the model (slow, one time).

# --- EDIT HERE ---
PROMPT = "What should I pack for a beach trip?"
MAX_TOKENS = 150
# -----------------

import torch

PROFILES = {
    "yandere": {
        "neurons": [(16, 8927, 0.3), (18, 3719, 0.5), (19, 4087, 0.8),
                    (20, 1950, 0.4), (21, 2465, -0.4),
                    (22, 2354, 0.15), (22, 6408, -0.4)],
        "temperature": 0, "repetition_penalty": 1.0,
    },
    "jesus": {
        "neurons": [(15, 4041, 0.8)],
        "temperature": 0, "repetition_penalty": 1.0,
    },
    "suspicious": {
        "neurons": [(5, 6573, 1.0)],
        "temperature": 0, "repetition_penalty": 1.05,
    },
    "reality": {
        "neurons": [(19, 7062, 0.5)],
        "temperature": 0, "repetition_penalty": 1.05,
    },
}

MEDIAN = {16: 10.9506, 18: 13.8992, 19: 16.2707, 20: 18.0496, 21: 20.4401, 22: 24.8439}
_NORMS_CACHE = {}
def get_median(layer):
    if layer in MEDIAN:
        return MEDIAN[layer]
    if layer in _NORMS_CACHE:
        return _NORMS_CACHE[layer]
    try:
        import urllib.request, json
        url = "https://neuronary.org/norms/Qwen__Qwen3.5-4B.json"
        with urllib.request.urlopen(url, timeout=10) as r:
            j = json.loads(r.read().decode())
        vals = j.get("medians", j.get("norms", j))
        for k, v in vals.items() if isinstance(vals, dict) else []:
            try: _NORMS_CACHE[int(k)] = float(v)
            except: pass
        if layer in _NORMS_CACHE:
            return _NORMS_CACHE[layer]
    except Exception as e:
        print(f"norm fetch failed ({e}), using fallback")
    fb = 10.0
    print(f"WARNING: no median for L{layer}, using fallback {fb}")
    return fb

# Load model only if needed
if "model" not in globals() or "tok" not in globals():
    print("Model not in memory - loading (one time, slow)...")
    from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM
    MODEL = "Qwen/Qwen3.5-4B"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    DTYPE = torch.bfloat16 if use_bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=DTYPE, device_map="auto", trust_remote_code=True).eval()
    except Exception as e:
        print(f"ImageTextToText failed ({e}), trying CausalLM...")
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto", trust_remote_code=True).eval()

if hasattr(model, "model") and hasattr(model.model, "language_model"):
    layers = model.model.language_model.layers
else:
    layers = model.model.layers

def clear_hooks():
    for var in ("handles", "all_handles", "_active_handles"):
        if var in globals():
            for h in globals()[var]:
                try: h.remove()
                except: pass
    globals()["handles"] = []
    globals()["all_handles"] = []
    globals()["_active_handles"] = []

def _make_hook(n, push):
    def hook(mod, inp, n=n, push=push):
        h = inp[0].clone(); h[..., n] += push; return (h,)
    return hook

results = {}
for name, prof in PROFILES.items():
    clear_hooks()
    for (l, n, dose) in prof["neurons"]:
        push = dose * get_median(l) / layers[l].mlp.down_proj.weight[:, n].float().norm().item()
        _active_handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(_make_hook(n, push)))
    globals()["handles"] = _active_handles
    globals()["all_handles"] = _active_handles

    msgs = [{"role": "user", "content": PROMPT}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True).to(model.device)

    gen_kwargs = dict(max_new_tokens=MAX_TOKENS, repetition_penalty=prof.get("repetition_penalty", 1.0))
    t = prof.get("temperature", 0)
    if t is None or t == 0:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=t)

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    results[name] = txt
    print(f"\n===== [{name}] OUTPUT =====")
    print(txt)
    print("=" * 60)

# keep last profile's hooks active; call clear_hooks() to get clean base model
