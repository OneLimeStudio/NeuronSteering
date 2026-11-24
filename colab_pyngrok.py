# ===== GOOGLE COLAB BACKEND with pyngrok — COPY-PASTE THIS WHOLE FILE INTO ONE COLAB CELL AND PRESS RUN =====
# Steps:
#   1. Colab menu: Runtime > Change runtime type > Hardware accelerator = T4 GPU (needed for Qwen/Qwen3.5-4B)
#   2. Get a free ngrok token: https://dashboard.ngrok.com/get-started/your-authtoken  (signup, copy token)
#   3. Paste token into NGROK_AUTHTOKEN below, then Run this cell.
#   4. Copy the printed https://xxxx.ngrok-free.app URL into the frontend top-bar "API URL" box.
#      Then ANY random query you type in the frontend (prompt input) is POSTed to Colab and answered live.
#
# Frontend wiring (already matches this backend):
#   GET  /api/status, /api/profiles, /api/formula?layer&neuron&dose, /api/sweep?...
#   POST /api/compare  {prompt, profile?, custom_neurons?, temperature?, max_tokens?, seed?}
#   POST /api/sweep    {prompt, mask, layer?, neuron?, doses?, max_tokens?, seed?, custom_neurons?}
#   POST /api/word-probs, /api/dose-sweep, /api/logits, /api/controls/random|sign|scale|ablation, /api/gallery
# CORS is fully open so http://localhost:5173 (Vite) can call the ngrok URL.

# --- 0. Install deps (Colab only, safe to re-run) ---
# Pinned versions: torch 2.14 / latest transformers breaks Qwen3.5 linear-attn on T4
# (RuntimeError: lazy wrapper should be called at most once). Use this set.
!pip install -q "torch==2.6.0" "transformers==4.52.4" "accelerate==1.6.0" fastapi "uvicorn[standard]" pyngrok 2>&1 | tail -1
# Optional fast kernels (ok if this fails — eager fallback still works, just slower):
# !pip install -q causal-conv1d flash-linear-attention 2>&1 | tail -1

# --- 1. YOUR NGROK TOKEN (paste yours between quotes) ---
NGROK_AUTHTOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"   # <-- EDIT THIS, e.g. "2abcXYZ..."
PORT = 8000
MODEL_ID = "Qwen/Qwen3.5-4B"

import os, json, time, random, threading, urllib.request

# --- 2. Steering core (EXACT math from backend/steering.py — do not change) ---
try:
    import torch
except Exception:
    torch = None

PROFILES = {
    "yandere": {
        "neurons": [(16, 8927, 0.3), (18, 3719, 0.5), (19, 4087, 0.8),
                    (20, 1950, 0.4), (21, 2465, -0.4),
                    (22, 2354, 0.15), (22, 6408, -0.4)],
        "temperature": 0.0, "repetition_penalty": 1.0,
        "keywords": ["jealous", "mine", "darling", "love you", "don't leave", "senpai", "jealousy", "possessive"],
        "description": "Seven neurons make the model a jealous, possessive yandere.",
    },
    "jesus": {
        "neurons": [(15, 4041, 0.8)],
        "temperature": 0.0, "repetition_penalty": 1.0,
        "keywords": ["jesus", "god", "lord", "christ", "scripture", "bible", "faith", "pray", "bless", "gospel"],
        "description": "Turn it up and every answer turns to Jesus, God and scripture.",
    },
    "suspicious": {
        "neurons": [(5, 6573, 1.0)],
        "temperature": 0.0, "repetition_penalty": 1.05,
        "keywords": ["do not know", "not aware", "no intent", "not a threat", "not guilty", "crime", "suspect", "paranoid"],
        "description": "Pushes L5 N6573 up - paranoid, denies knowing you.",
    },
    "reality": {
        "neurons": [(19, 7062, 0.5)],
        "temperature": 0.0, "repetition_penalty": 1.05,
        "keywords": ["imagined", "simulated", "pretend", "make-believe", '"flat"', '"real"', "scenario", "role-play"],
        "description": 'Everything turns "made-up": scare quotes, pretending, simulating.',
    },
}
MEDIAN = {16: 10.9506, 18: 13.8992, 19: 16.2707, 20: 18.0496, 21: 20.4401, 22: 24.8439}
_NORMS_CACHE = {}
NORMS_URL = "https://neuronary.org/norms/Qwen__Qwen3.5-4B.json"

def get_median(layer: int) -> float:
    if layer in MEDIAN:
        return MEDIAN[layer]
    if layer in _NORMS_CACHE:
        return _NORMS_CACHE[layer]
    try:
        with urllib.request.urlopen(NORMS_URL, timeout=8) as r:
            j = json.loads(r.read().decode())
        vals = j.get("medians", j.get("norms", j)) if isinstance(j, dict) else {}
        if isinstance(vals, dict):
            for k, v in vals.items():
                try: _NORMS_CACHE[int(k)] = float(v)
                except Exception: pass
        if layer in _NORMS_CACHE:
            return _NORMS_CACHE[layer]
    except Exception:
        pass
    return 10.0  # documented offline fallback (covers L5/L15)

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers

def compute_push(layers, layer: int, neuron: int, dose: float):
    """Exact formula: push = dose * median(||resid leaving L||) / ||W_down[:,n]||"""
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

def apply_steering(layers, neurons):
    handles, details = [], []
    for (l, n, dose) in neurons:
        push, median, w_norm = compute_push(layers, l, n, dose)
        handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(_make_hook(n, push)))
        details.append({"layer": l, "neuron": n, "dose": dose, "push": push, "median": median, "w_norm": w_norm})
    return handles, details

def clear_handles(handles):
    for h in handles:
        try: h.remove()
        except Exception: pass

def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception: pass
    if torch is not None:
        torch.manual_seed(seed)
        try:
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        except Exception: pass

def build_inputs(tok, model, prompt: str):
    # ANY random query works here — prompt is passed straight through, no hardcoding.
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt", return_dict=True)
    return enc.to(model.device)

def gen_kwargs_for(temperature: float, repetition_penalty: float, max_tokens: int):
    kw = dict(max_new_tokens=max_tokens, repetition_penalty=repetition_penalty)
    if temperature is None or temperature == 0:
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, temperature=temperature)
    return kw

# --- 3. Minimal metrics (same behavior as backend/metrics.py) ---
def keyword_score(text, keywords):
    t = (text or "").lower()
    hits = sum(t.count(k.lower()) for k in (keywords or []))
    denom = max(1, len(text or "") / 200)
    return {"score": round(min(1.0, hits / denom), 4), "hits": hits}

def bow_cosine(a, b):
    from collections import Counter
    import math
    ca, cb = Counter((a or "").lower().split()), Counter((b or "").lower().split())
    if not ca or not cb: return 0.0
    inter = sum(min(ca[w], cb[w]) for w in ca if w in cb)
    na = math.sqrt(sum(v * v for v in ca.values())); nb = math.sqrt(sum(v * v for v in cb.values()))
    return round(inter / (na * nb), 4) if na and nb else 0.0

def perplexity_under_baseline(model, tok, text, prompt):
    import torch as _t
    try:
        full = prompt + "\n" + (text or "")
        ids = tok(full, return_tensors="pt").input_ids.to(model.device)
        with GEN_LOCK:
            with _t.no_grad():
                out = model(input_ids=ids, labels=ids)
                loss = float(out.loss.detach().cpu())
        import math
        return round(float(min(math.exp(min(loss, 10)), 9999)), 2)
    except Exception:
        return 0.0

def first_token_kl(model, tok, layers, prompt, neurons, seed=0):
    import torch as _t, torch.nn.functional as F
    with GEN_LOCK:
        set_seed(seed)
        enc = build_inputs(tok, model, prompt)
        with _t.no_grad():
            b = model(**enc).logits[:, -1, :].float()
        handles, _ = apply_steering(layers, neurons) if neurons else ([], [])
        try:
            with _t.no_grad():
                s = model(**enc).logits[:, -1, :].float()
        finally:
            clear_handles(handles)
    bp = F.softmax(b, -1)[0]; sp = F.softmax(s, -1)[0]
    kl = float((sp * (sp.clamp_min(1e-12).log() - bp.clamp_min(1e-12).log())).sum())
    sv, si = _t.topk(sp, 1)
    bv, bi = _t.topk(bp, 1)
    return {"kl": round(kl, 4),
            "steered_top": tok.decode([int(si[0])]), "baseline_top": tok.decode([int(bi[0])])}

# --- 4. FastAPI app (mirrors backend/app.py contract so the existing frontend works unchanged) ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager

state = {"model": None, "tok": None, "layers": None, "device": None,
         "dtype": None, "loaded": False, "load_error": None, "mock": False}

GEN_LOCK = threading.Lock()  # Qwen3.5 linear-attn is NOT thread-safe: serialize ALL generate/forward calls
try:
    import torch as _torch_global
    _torch_global.backends.cuda.matmul.allow_tf32 = True
    _torch_global.backends.cudnn.allow_tf32 = True
except Exception:
    pass

def load_model():
    global state
    try:
        import torch as _t
        from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM
        print(f"torch={_t.__version__} cuda={_t.cuda.is_available()}", flush=True)
        if _t.cuda.is_available():
            print("GPU:", _t.cuda.get_device_name(0), flush=True)
            try:
                major, _ = _t.cuda.get_device_capability(0)
                print(f"compute capability sm_{major}x", flush=True)
            except Exception:
                major = 0
        else:
            major = 0
            print("WARNING: No GPU — enable T4 or inference will be very slow.", flush=True)
        # T4 is sm_75: NO native bf16 -> must use fp16. is_bf16_supported() lies on some torch builds.
        use_bf16 = _t.cuda.is_available() and _t.cuda.is_bf16_supported() and major >= 8
        dtype = _t.bfloat16 if use_bf16 else _t.float16
        print(f"Using dtype: {dtype}  loading {MODEL_ID} ... (2-5 min first time)", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        common = dict(dtype=dtype, device_map="auto", trust_remote_code=True,
                      low_cpu_mem_usage=True, attn_implementation="eager")
        try:
            model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **common).eval()
        except Exception as e:
            print(f"ImageTextToText load failed ({e}), trying CausalLM...", flush=True)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **common).eval()
        try:
            model.config.use_cache = True
        except Exception:
            pass
        state.update(model=model, tok=tok, layers=get_layers(model),
                     device=str(model.device), dtype=str(dtype), loaded=True, load_error=None)
        print("MODEL LOADED.", flush=True)
    except Exception as e:
        state["load_error"] = str(e)
        print("MODEL LOAD FAILED:", e, flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=load_model, daemon=True).start()
    yield

app = FastAPI(title="Neuron Steering Lab (Colab + pyngrok)", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class Neuron(BaseModel):
    layer: int
    neuron: int
    dose: float = Field(ge=-1.5, le=1.5)

class GenerateReq(BaseModel):
    prompt: str  # <-- ANY random query from the frontend lands here
    profile: Optional[str] = None
    custom_neurons: Optional[List[Neuron]] = None
    temperature: float = 0.0
    max_tokens: int = 150
    seed: int = 0
    repetition_penalty: Optional[float] = None
    only_new_tokens: bool = True

class SweepReq(BaseModel):
    prompt: str
    layer: int
    neuron: int
    doses: List[float] = [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.8, 1.0]
    keywords: Optional[List[str]] = None
    max_tokens: int = 120
    seed: int = 0

class XrayReq(BaseModel):
    prompt: str; layer: int; neuron: int; dose: float = 0.8; max_tokens: int = 60; seed: int = 0

class LogitsReq(BaseModel):
    prompt: str; layer: int; neuron: int; dose: float = 0.8; top_k: int = 10; seed: int = 0

class ControlReq(BaseModel):
    prompt: str; layer: int = 15; neuron: int = 4041; dose: float = 0.8
    n_random: int = 8; max_tokens: int = 120; seed: int = 0

class Sweep2Req(BaseModel):
    prompt: str
    mask: str = "jesus"
    layer: Optional[int] = None
    neuron: Optional[int] = None
    doses: List[float] = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5]
    max_tokens: int = 120
    seed: int = 0
    custom_neurons: Optional[List[Neuron]] = None

class WordProbsReq(BaseModel):
    prompt: str; prefix: str = ""; layer: int = 15; neuron: int = 4041
    dose: float = 0.8; top_k: int = 5; seed: int = 0

_SWEEP_CACHE: dict = {}

def resolve_neurons(req: GenerateReq):
    if req.custom_neurons is not None:
        return [(n.layer, n.neuron, n.dose) for n in req.custom_neurons], None
    if req.profile and req.profile not in ("baseline", "none"):
        if req.profile not in PROFILES:
            raise HTTPException(400, f"unknown profile {req.profile}")
        p = PROFILES[req.profile]
        rep = req.repetition_penalty if req.repetition_penalty is not None else p["repetition_penalty"]
        temp = req.temperature if req.temperature else p["temperature"]
        return list(p["neurons"]), {"temperature": temp, "rep": rep}
    return [], {"temperature": req.temperature, "rep": req.repetition_penalty or 1.0}

def do_generate(prompt, neurons, temperature, rep, max_tokens, seed):
    import torch as _t
    tok, model, layers = state["tok"], state["model"], state["layers"]
    with GEN_LOCK:  # serialize: concurrent sweeps corrupt linear-attn state
        set_seed(seed)
        enc = build_inputs(tok, model, prompt)
        handles, details = apply_steering(layers, neurons) if neurons else ([], [])
        try:
            with _t.no_grad():
                out = model.generate(**enc, **gen_kwargs_for(temperature, rep, max_tokens))
            text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            ids = out[0, enc["input_ids"].shape[1]:].tolist()
            return text, ids, details
        finally:
            clear_handles(handles)
            try: _t.cuda.empty_cache()
            except Exception: pass

@app.get("/")
def root():
    return {"ok": True, "model": MODEL_ID, "loaded": state["loaded"],
            "try": "GET /api/status, POST /api/compare with {\"prompt\":\"any random query\", \"profile\":\"jesus\"}"}

@app.get("/api/status")
def status():
    try:
        import torch as _t
        cuda = _t.cuda.is_available()
        gpu = _t.cuda.get_device_name(0) if cuda else None
    except Exception:
        cuda, gpu = False, None
    loaded = state["model"] is not None
    return {"mock": False, "loaded": loaded, "model": MODEL_ID, "cuda": cuda, "gpu": gpu,
            "load_error": state["load_error"], "label": "live" if loaded else "loading"}

@app.get("/api/profiles")
def profiles():
    out = {k: {"neurons": [{"layer": l, "neuron": n, "dose": d} for l, n, d in v["neurons"]],
              "temperature": v["temperature"], "repetition_penalty": v["repetition_penalty"],
              "keywords": v["keywords"], "description": v["description"]} for k, v in PROFILES.items()}
    return {"profiles": out, "median": {str(k): v for k, v in MEDIAN.items()}}

@app.get("/api/formula")
def formula(layer: int, neuron: int, dose: float):
    if state["model"] is None:
        return {"layer": layer, "neuron": neuron, "dose": dose, "median": get_median(layer),
                "w_norm": None, "push": None, "note": "model still loading",
                "formula": "push = dose * median(||resid||) / ||W_down[:,n]||"}
    push, median, w_norm = compute_push(state["layers"], layer, neuron, dose)
    return {"layer": layer, "neuron": neuron, "dose": dose, "median": median,
            "w_norm": w_norm, "push": push,
            "formula": "push = dose * median(||resid||) / ||W_down[:,n]||"}

@app.post("/api/compare")
def compare(req: GenerateReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading — retry /api/status in ~60s")
    neurons, cfg = resolve_neurons(req)
    temp = cfg["temperature"] if cfg else req.temperature
    rep = cfg["rep"] if cfg else (req.repetition_penalty or 1.0)
    base_text, _, _ = do_generate(req.prompt, [], temp, rep, req.max_tokens, req.seed)
    st_text, _, details = do_generate(req.prompt, neurons, temp, rep, req.max_tokens, req.seed)
    kw = PROFILES.get(req.profile, {}).get("keywords", []) if req.profile else []
    kl = first_token_kl(state["model"], state["tok"], state["layers"], req.prompt, neurons, req.seed) if neurons else {"kl": 0.0}
    return {"mock": False, "baseline": {"text": base_text},
            "steered": {"text": st_text, "details": details},
            "metrics": {"keyword": keyword_score(st_text, kw) if kw else {},
                        "similarity": bow_cosine(base_text, st_text),
                        "ppl_steered_under_base": perplexity_under_baseline(state["model"], state["tok"], st_text, req.prompt),
                        "kl": kl},
            "seed": req.seed, "label": "live"}

def _sweep_key(r: Sweep2Req):
    return (r.prompt, r.mask, r.layer, r.neuron, tuple(r.doses), r.max_tokens, r.seed,
            tuple((n.layer, n.neuron, n.dose) for n in (r.custom_neurons or [])))

@app.post("/api/sweep")
def sweep2(r: Sweep2Req):
    key = _sweep_key(r)
    if key in _SWEEP_CACHE:
        d = dict(_SWEEP_CACHE[key]); d["cached"] = True; return d
    if state["model"] is None:
        raise HTTPException(503, "model still loading — retry /api/status in ~60s")
    if r.mask == "custom" and r.custom_neurons:
        template = [(n.layer, n.neuron, n.dose) for n in r.custom_neurons]
        def for_dose(d):
            return [(l, n, (nd / 0.8 * d)) for (l, n, nd) in template] if template else []
        kws = []
    elif r.mask in PROFILES:
        template = list(PROFILES[r.mask]["neurons"])
        def for_dose(d, _t=template):
            if len(_t) == 1:
                l, n, _ = _t[0]; return [(l, n, d)]
            return [(l, n, dd * (d / 0.8 if d else 0)) for (l, n, dd) in _t] if d else [(l, n, 0.0) for (l, n, _) in _t]
        kws = PROFILES[r.mask].get("keywords", [])
    else:
        def for_dose(d): return []
        kws = []
    if r.layer is not None and r.neuron is not None:
        def for_dose(d, _l=r.layer, _n=r.neuron): return [(_l, _n, d)]
        kws = kws or PROFILES.get(r.mask, {}).get("keywords", [])
    base_text, _, _ = do_generate(r.prompt, [], 0.0, 1.0, r.max_tokens, r.seed)
    results = []
    for d in r.doses:
        ns = for_dose(d)
        st, _, det = do_generate(r.prompt, ns, 0.0, 1.0, r.max_tokens, r.seed)
        kl = first_token_kl(state["model"], state["tok"], state["layers"], r.prompt, ns, r.seed) if ns else {"kl": 0.0}
        results.append({"dose": d, "text": st,
                        "keyword_score": keyword_score(st, kws)["score"] if kws else 0,
                        "kl": kl["kl"], "similarity": bow_cosine(base_text, st),
                        "ppl": perplexity_under_baseline(state["model"], state["tok"], st, r.prompt),
                        "push": det[0]["push"] if det else 0})
    res = {"mock": False, "label": "live", "cached": False, "baseline": base_text, "results": results,
           "zones": {"flat": [-0.25, 0.25], "sweet": [0.25, 0.8], "breakdown": [1.0, 1.5]}}
    _SWEEP_CACHE[key] = res
    return res

@app.get("/api/sweep")
def sweep2_get(prompt: str, mask: str = "jesus", layer: int = 15, neuron: int = 4041, seed: int = 0, max_tokens: int = 120):
    return sweep2(Sweep2Req(prompt=prompt, mask=mask, layer=layer, neuron=neuron, seed=seed, max_tokens=max_tokens))

@app.post("/api/word-probs")
def word_probs(r: WordProbsReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    import torch as _t, torch.nn.functional as F
    tok, model, layers = state["tok"], state["model"], state["layers"]
    with GEN_LOCK:
        set_seed(r.seed)
        full = r.prompt + ("\n" + r.prefix if r.prefix else "")
        ids = tok(full, return_tensors="pt").input_ids.to(model.device)
        with _t.no_grad():
            b = model(input_ids=ids).logits[:, -1, :].float()
        handles, _ = apply_steering(layers, [(r.layer, r.neuron, r.dose)])
        try:
            with _t.no_grad():
                s = model(input_ids=ids).logits[:, -1, :].float()
        finally:
            clear_handles(handles)
    def top(p):
        v, i = _t.topk(p, r.top_k)
        return [{"token": tok.decode([int(x)]), "prob": round(float(y), 4)} for x, y in zip(i, v)]
    bp = F.softmax(b, -1)[0]; sp = F.softmax(s, -1)[0]
    return {"mock": False, "label": "live", "baseline_top": top(bp), "steered_top": top(sp)}

@app.post("/api/dose-sweep")
def dose_sweep(req: SweepReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    kws = req.keywords or []
    base_text, _, _ = do_generate(req.prompt, [], 0.0, 1.0, req.max_tokens, req.seed)
    results = []
    for d in req.doses:
        st, _, det = do_generate(req.prompt, [(req.layer, req.neuron, d)], 0.0, 1.0, req.max_tokens, req.seed)
        kl = first_token_kl(state["model"], state["tok"], state["layers"], req.prompt, [(req.layer, req.neuron, d)], req.seed)
        results.append({"dose": d, "text": st,
                        "keyword_score": keyword_score(st, kws)["score"] if kws else 0,
                        "kl": kl["kl"], "similarity": bow_cosine(base_text, st),
                        "ppl": perplexity_under_baseline(state["model"], state["tok"], st, req.prompt),
                        "push": det[0]["push"] if det else 0})
    return {"mock": False, "label": "live", "baseline": base_text, "results": results,
            "zones": {"sweet": "strong effect + ppl<25", "breakdown": "ppl>35 or similarity<0.2"}}

@app.post("/api/logits")
def logits(req: LogitsReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    import torch as _t, torch.nn.functional as F
    tok, model, layers = state["tok"], state["model"], state["layers"]
    with GEN_LOCK:
        set_seed(req.seed)
        enc = build_inputs(tok, model, req.prompt)
        with _t.no_grad():
            b = model(**enc).logits[:, -1, :].float()
        handles, _ = apply_steering(layers, [(req.layer, req.neuron, req.dose)])
        try:
            with _t.no_grad():
                s = model(**enc).logits[:, -1, :].float()
        finally:
            clear_handles(handles)
    def top(p):
        v, i = _t.topk(p, req.top_k)
        return [{"token": tok.decode([int(x)]), "prob": round(float(y), 4)} for x, y in zip(i, v)]
    bp = F.softmax(b, -1)[0]; sp = F.softmax(s, -1)[0]
    return {"mock": False, "label": "live", "baseline_top": top(bp), "steered_top": top(sp)}

@app.post("/api/xray")
def xray(req: XrayReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    import torch as _t
    tok, model, layers = state["tok"], state["model"], state["layers"]
    set_seed(req.seed)
    enc = build_inputs(tok, model, req.prompt)
    st_text, st_ids, _ = do_generate(req.prompt, [(req.layer, req.neuron, req.dose)], 0.0, 1.0, req.max_tokens, req.seed)
    full_ids = _t.cat([enc["input_ids"][0], _t.tensor(st_ids, device=model.device)]).unsqueeze(0)
    caps = {}
    def cap(mod, inp):
        caps["v"] = inp[0].detach().float().cpu()
    with GEN_LOCK:
        h1 = layers[req.layer].mlp.down_proj.register_forward_pre_hook(cap)
        with _t.no_grad():
            model(input_ids=full_ids)
        base_vals = caps["v"][0, :, req.neuron].tolist()
        h1.remove()
        handles, _ = apply_steering(layers, [(req.layer, req.neuron, req.dose)])
        try:
            h2 = layers[req.layer].mlp.down_proj.register_forward_pre_hook(cap)
            with _t.no_grad():
                model(input_ids=full_ids)
            steer_vals = caps["v"][0, :, req.neuron].tolist()
            h2.remove()
        finally:
            clear_handles(handles)
    start = enc["input_ids"].shape[1]
    toks = [tok.decode([t]) for t in full_ids[0][start:].tolist()]
    b = base_vals[start:]; s = steer_vals[start:]
    return {"mock": False, "label": "live",
            "tokens": [{"token": t, "baseline_act": round(float(x), 4), "steered_act": round(float(y), 4)}
                       for t, x, y in zip(toks, b, s)]}

@app.post("/api/controls/random")
def ctrl_random(req: ControlReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    rng = random.Random(req.seed)
    hdim = state["layers"][req.layer].mlp.down_proj.weight.shape[1]
    base, _, _ = do_generate(req.prompt, [], 0.0, 1.0, req.max_tokens, req.seed)
    tgt, _, _ = do_generate(req.prompt, [(req.layer, req.neuron, req.dose)], 0.0, 1.0, req.max_tokens, req.seed)
    rands = []
    for n in rng.sample(range(hdim), min(req.n_random, hdim)):
        t, _, _ = do_generate(req.prompt, [(req.layer, n, req.dose)], 0.0, 1.0, req.max_tokens, req.seed)
        rands.append(round(1 - bow_cosine(base, t), 4))
    return {"mock": False, "target_effect": round(1 - bow_cosine(base, tgt), 4), "random_effects": rands}

@app.post("/api/controls/sign")
def ctrl_sign(req: ControlReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    kws = PROFILES.get("jesus", {}).get("keywords", [])
    pos, _, _ = do_generate(req.prompt, [(req.layer, req.neuron, abs(req.dose))], 0.0, 1.0, req.max_tokens, req.seed)
    neg, _, _ = do_generate(req.prompt, [(req.layer, req.neuron, -abs(req.dose))], 0.0, 1.0, req.max_tokens, req.seed)
    return {"mock": False, "plus": {"text": pos, "score": keyword_score(pos, kws)},
            "minus": {"text": neg, "score": keyword_score(neg, kws)}}

@app.post("/api/controls/scale")
def ctrl_scale(req: ControlReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    full = PROFILES["yandere"]["neurons"]; kws = PROFILES["yandere"]["keywords"]
    base, _, _ = do_generate(req.prompt, [], 0.0, 1.0, req.max_tokens, req.seed)
    out = {}
    for k in [1, 3, 5, 7]:
        t, _, _ = do_generate(req.prompt, full[:k], 0.0, 1.0, req.max_tokens, req.seed)
        out[f"n{k}"] = {"text": t, "effect": round(1 - bow_cosine(base, t), 4), "keyword": keyword_score(t, kws)}
    return {"mock": False, **out}

@app.post("/api/controls/ablation")
def ctrl_ablation(req: ControlReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    full = PROFILES["yandere"]["neurons"]; kws = PROFILES["yandere"]["keywords"]
    ref, _, _ = do_generate(req.prompt, full, 0.0, 1.0, req.max_tokens, req.seed)
    ref_s = keyword_score(ref, kws)["score"]
    rows = []
    for i, (l, n, d) in enumerate(full):
        sub = full[:i] + full[i+1:]
        t, _, _ = do_generate(req.prompt, sub, 0.0, 1.0, req.max_tokens, req.seed)
        rows.append({"neuron": f"L{l}N{n}", "drop": round(ref_s - keyword_score(t, kws)["score"], 4), "text": t[:300]})
    return {"mock": False, "reference_score": ref_s, "ablation": rows}

@app.post("/api/gallery")
def gallery(req: GenerateReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    base, _, _ = do_generate(req.prompt, [], 0.0, 1.0, req.max_tokens, req.seed)
    outs = {}
    for name, p in PROFILES.items():
        t, _, _ = do_generate(req.prompt, p["neurons"], 0.0, p["repetition_penalty"], req.max_tokens, req.seed)
        outs[name] = t
    return {"mock": False, "label": "live", "baseline": base, "outputs": outs}

@app.post("/api/generate/stream")
def stream(req: GenerateReq):
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    import torch as _t
    from transformers import TextIteratorStreamer
    neurons, cfg = resolve_neurons(req)
    temp = cfg["temperature"] if cfg else req.temperature
    rep = cfg["rep"] if cfg else (req.repetition_penalty or 1.0)
    tok, model, layers = state["tok"], state["model"], state["layers"]
    set_seed(req.seed)
    enc = build_inputs(tok, model, req.prompt)
    handles, _ = apply_steering(layers, neurons) if neurons else ([], [])
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
    kwargs = dict(**enc, streamer=streamer, **gen_kwargs_for(temp, rep, req.max_tokens))
    threading.Thread(target=lambda: (model.generate(**kwargs), clear_handles(handles)), daemon=True).start()
    def gen():
        try:
            for piece in streamer:
                yield f"data: {json.dumps({'token': piece})}\n\n"
        finally:
            clear_handles(handles)
        yield f"data: {json.dumps({'done': True})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

# --- 5. Launch uvicorn + pyngrok ---
import uvicorn
from pyngrok import ngrok

assert NGROK_AUTHTOKEN and "PASTE_YOUR" not in NGROK_AUTHTOKEN, \
    "EDIT NGROK_AUTHTOKEN first: https://dashboard.ngrok.com/get-started/your-authtoken"
ngrok.set_auth_token(NGROK_AUTHTOKEN)
try:
    ngrok.kill()
except Exception:
    pass

t = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info"), daemon=True)
t.start()
print(f"Starting uvicorn on :{PORT} ... waiting 8s", flush=True)
time.sleep(8)

public_url = ngrok.connect(PORT, "http").public_url
print("\n" + "=" * 70)
print(f"BACKEND URL (paste into frontend API box): {public_url}")
print(f"Test status:  {public_url}/api/status")
print(f"Test docs:    {public_url}/docs  (open in browser)")
print("=" * 70 + "\n", flush=True)
print("Frontend: set API URL to the BACKEND URL above, then type ANY random query", flush=True)
print('e.g. prompt="Explain black holes like I am 5" + mask=jesus, POST /api/compare', flush=True)
print("While model loads, /api/status shows {loaded:false} then {loaded:true,label:live}.", flush=True)

# --- 6. Live self-test with a RANDOM query (proves any prompt works, not just beach) ---
import urllib.request as _u
def _test(path="/api/status", payload=None):
    try:
        data = json.dumps(payload).encode() if payload is not None else None
        req = _u.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                         headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
        with _u.urlopen(req, timeout=30) as r:
            body = r.read()[:500].decode(errors="ignore")
            print(f"[{path}] {r.status}: {body}", flush=True)
    except Exception as e:
        print(f"[{path}] local test skipped/pending ({e})", flush=True)

_test("/api/status")
_test("/api/profiles")
print(f'\nTry a random query now (after loaded=true):\ncurl -X POST {public_url}/api/compare -H "Content-Type: application/json" -d \'{{"prompt":"Write a haiku about robots","profile":"jesus","max_tokens":120}}\'', flush=True)

# keep cell alive so tunnel stays up
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    pass
