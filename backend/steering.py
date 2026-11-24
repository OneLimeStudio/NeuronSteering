"""Steering core — REUSED EXACTLY from main.py / ask_one_profile.py. DO NOT change math."""
import random
from typing import List, Tuple

try:
    import torch  # optional in MOCK_MODE
except Exception:
    torch = None  # type: ignore

MODEL_ID = "Qwen/Qwen3.5-4B"

# Exact profiles from spec (editable from UI via custom neurons, values locked here)
PROFILES = {
    "yandere": {
        "neurons": [(16, 8927, 0.3), (18, 3719, 0.5), (19, 4087, 0.8),
                    (20, 1950, 0.4), (21, 2465, -0.4),
                    (22, 2354, 0.15), (22, 6408, -0.4)],
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "keywords": ["jealous", "mine", "darling", "love you", "don't leave", "senpai", "jealousy", "possessive"],
        "description": "Seven neurons make the model a jealous, possessive yandere.",
    },
    "jesus": {
        "neurons": [(15, 4041, 0.8)],
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "keywords": ["jesus", "god", "lord", "christ", "scripture", "bible", "faith", "pray", "bless", "gospel"],
        "description": "Turn it up and every answer turns to Jesus, God and scripture.",
    },
    "suspicious": {
        "neurons": [(5, 6573, 1.0)],
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "keywords": ["do not know", "not aware", "no intent", "not a threat", "not guilty", "crime", "suspect", "paranoid"],
        "description": "Pushes L5 N6573 up — paranoid, denies knowing you.",
    },
    "reality": {
        "neurons": [(19, 7062, 0.5)],
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "keywords": ["imagined", "simulated", "pretend", "make-believe", '"flat"', '"real"', "scenario", "role-play"],
        "description": 'Everything turns "made-up": scare quotes, pretending, simulating.',
    },
}

# median ||residual leaving layer L|| from neuronary.org/norms/Qwen__Qwen3.5-4B.json
MEDIAN = {16: 10.9506, 18: 13.8992, 19: 16.2707, 20: 18.0496, 21: 20.4401, 22: 24.8439}
_NORMS_CACHE: dict = {}
NORMS_URL = "https://neuronary.org/norms/Qwen__Qwen3.5-4B.json"


def get_median(layer: int) -> float:
    if layer in MEDIAN:
        return MEDIAN[layer]
    if layer in _NORMS_CACHE:
        return _NORMS_CACHE[layer]
    try:
        import urllib.request, json
        with urllib.request.urlopen(NORMS_URL, timeout=8) as r:
            j = json.loads(r.read().decode())
        vals = j.get("medians", j.get("norms", j)) if isinstance(j, dict) else {}
        if isinstance(vals, dict):
            for k, v in vals.items():
                try:
                    _NORMS_CACHE[int(k)] = float(v)
                except Exception:
                    pass
        if layer in _NORMS_CACHE:
            return _NORMS_CACHE[layer]
    except Exception:
        pass
    return 10.0  # documented fallback for L5/L15 when offline


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def compute_push(layers, layer: int, neuron: int, dose: float):
    """Exact formula: push = dose * median / ||W_down[:, n]||"""
    median = get_median(layer)
    w_norm = layers[layer].mlp.down_proj.weight[:, neuron].float().norm().item()
    push = dose * median / w_norm
    return push, median, w_norm


def _make_hook(neuron: int, push: float):
    def hook(mod, inp, n=neuron, push=push):
        h = inp[0].clone()
        h[..., n] += push
        return (h,)
    return hook


def apply_steering(layers, neurons: List[Tuple[int, int, float]]):
    """Apply hooks. Returns (handles, details). Caller MUST remove handles in finally."""
    handles = []
    details = []
    for (l, n, dose) in neurons:
        push, median, w_norm = compute_push(layers, l, n, dose)
        handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(_make_hook(n, push)))
        details.append({"layer": l, "neuron": n, "dose": dose, "push": push,
                        "median": median, "w_norm": w_norm})
    return handles, details


def clear_handles(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    if torch is None:
        return  # mock / torch-free environments: stdlib (+numpy) seeding only
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_inputs(tok, model, prompt: str):
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt", return_dict=True,
    )
    return enc.to(model.device)


def gen_kwargs_for(temperature: float, repetition_penalty: float, max_tokens: int):
    kw = dict(max_new_tokens=max_tokens, repetition_penalty=repetition_penalty)
    if temperature is None or temperature == 0:
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, temperature=temperature)
    return kw
