"""FastAPI backend — loads Qwen/Qwen3.5-4B ONCE, serves all lab endpoints. MOCK_MODE=1 for no-GPU demos."""
import os, time, json, random, threading
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MOCK_MODE = os.getenv("MOCK_MODE", "0") == "1"

from .steering import (
    PROFILES, MODEL_ID, get_median, get_layers, compute_push,
    apply_steering, clear_handles, set_seed, build_inputs, gen_kwargs_for,
)
from . import mock_data

state = {"model": None, "tok": None, "layers": None, "device": None, "dtype": None,
         "loaded": True if MOCK_MODE else False, "load_error": None, "mock": MOCK_MODE}


def load_model():
    if MOCK_MODE:
        state["loaded"] = True
        state["mock"] = True
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM
    try:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, dtype=dtype, device_map="auto", trust_remote_code=True).eval()
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=dtype, device_map="auto", trust_remote_code=True).eval()
        state.update(model=model, tok=tok, layers=get_layers(model),
                     device=str(model.device), dtype=str(dtype), loaded=True)
    except Exception as e:
        state["load_error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=load_model, daemon=True)
    t.start()
    yield

app = FastAPI(title="Neuron Steering Lab", version="1.0.0",
              description="Single-neuron steering demo API (Qwen3.5-4B). See /docs",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ---------- schemas ----------
class Neuron(BaseModel):
    layer: int
    neuron: int
    dose: float = Field(ge=-1.5, le=1.5)

class GenerateReq(BaseModel):
    prompt: str
    profile: Optional[str] = None  # yandere|jesus|suspicious|reality or None=baseline
    custom_neurons: Optional[List[Neuron]] = None
    temperature: float = 0.0
    max_tokens: int = 150
    seed: int = 0
    repetition_penalty: Optional[float] = None
    only_new_tokens: bool = True  # toggle kept for UI; steering always applies every token (paper method)

class SweepReq(BaseModel):
    prompt: str
    layer: int
    neuron: int
    doses: List[float] = [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.8, 1.0]
    keywords: Optional[List[str]] = None
    max_tokens: int = 120
    seed: int = 0

class XrayReq(BaseModel):
    prompt: str
    layer: int
    neuron: int
    dose: float = 0.8
    max_tokens: int = 60
    seed: int = 0

class LogitsReq(BaseModel):
    prompt: str
    layer: int
    neuron: int
    dose: float = 0.8
    top_k: int = 10
    seed: int = 0

class ControlReq(BaseModel):
    prompt: str
    layer: int = 15
    neuron: int = 4041
    dose: float = 0.8
    n_random: int = 8
    max_tokens: int = 120
    seed: int = 0


def resolve_neurons(req: GenerateReq):
    if req.custom_neurons is not None:
        return [(n.layer, n.neuron, n.dose) for n in req.custom_neurons], None
    if req.profile and req.profile != "baseline":
        if req.profile not in PROFILES:
            raise HTTPException(400, f"unknown profile {req.profile}")
        p = PROFILES[req.profile]
        rep = req.repetition_penalty if req.repetition_penalty is not None else p["repetition_penalty"]
        return list(p["neurons"]), {"temperature": req.temperature if req.temperature else p["temperature"], "rep": rep}
    return [], {"temperature": req.temperature, "rep": req.repetition_penalty or 1.0}


def do_generate(prompt, neurons, temperature, rep, max_tokens, seed):
    import torch
    tok, model, layers = state["tok"], state["model"], state["layers"]
    set_seed(seed)
    enc = build_inputs(tok, model, prompt)
    handles, details = apply_steering(layers, neurons) if neurons else ([], [])
    try:
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs_for(temperature, rep, max_tokens))
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ids = out[0, enc["input_ids"].shape[1]:].tolist()
        return text, ids, details
    finally:
        clear_handles(handles)


# ---------- endpoints ----------
@app.get("/api/status")
def status():
    try:
        import torch as _t
        cuda = _t.cuda.is_available() if not MOCK_MODE else False
        gpu = _t.cuda.get_device_name(0) if (not MOCK_MODE and cuda) else None
    except Exception:
        cuda, gpu = False, None
    return {"mock": state["mock"] or MOCK_MODE, "loaded": state["loaded"] if MOCK_MODE else (state["model"] is not None),
            "model": MODEL_ID, "cuda": cuda,
            "gpu": gpu,
            "load_error": state["load_error"], "label": "pre-recorded" if (state["mock"] or MOCK_MODE) else "live"}

@app.get("/api/profiles")
def profiles():
    out = {}
    for k, v in PROFILES.items():
        out[k] = {"neurons": [{"layer": l, "neuron": n, "dose": d} for l, n, d in v["neurons"]],
                  "temperature": v["temperature"], "repetition_penalty": v["repetition_penalty"],
                  "keywords": v["keywords"], "description": v["description"]}
    return {"profiles": out, "median": {str(k): v for k, v in
            __import__("backend.steering", fromlist=["MEDIAN"]).MEDIAN.items()}}

@app.get("/api/formula")
def formula(layer: int, neuron: int, dose: float):
    from .calibration import median_with_source
    median, source = median_with_source(layer)
    if MOCK_MODE or state["model"] is None:
        return {"layer": layer, "neuron": neuron, "dose": dose, "median": median,
                "median_source": source, "eligible": source != "fallback",
                "w_norm": None, "push": None, "note": "load model for live w_norm/push",
                "formula": "push = dose * median(||resid||) / ||W_down[:,n]||"}
    layers = state["layers"]
    push, median, w_norm = compute_push(layers, layer, neuron, dose)
    return {"layer": layer, "neuron": neuron, "dose": dose, "median": median,
            "median_source": source, "eligible": source != "fallback",
            "w_norm": w_norm, "push": push,
            "formula": "push = dose * median(||resid||) / ||W_down[:,n]||"}

@app.post("/api/compare")
def compare(req: GenerateReq):
    """Baseline vs steered side-by-side, deterministic (fixed seed)."""
    if MOCK_MODE:
        base = mock_data.BASELINE_BEACH if "beach" in req.prompt.lower() else f"(mock baseline) Echo: {req.prompt[:200]}"
        key = req.profile or "jesus"
        steered = mock_data.PROFILES_MOCK.get(key, f"(mock {key}) {req.prompt[:120]} …blessed.")
        return {"mock": True, "baseline": {"text": base}, "steered": {"text": steered, "profile": key},
                "seed": req.seed, "label": "pre-recorded"}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    from .metrics import keyword_score, bow_cosine, perplexity_under_baseline, first_token_kl
    neurons, cfg = resolve_neurons(req)
    temp = cfg["temperature"] if cfg else req.temperature
    rep = cfg["rep"] if cfg else (req.repetition_penalty or 1.0)
    base_text, base_ids, _ = do_generate(req.prompt, [], temp, rep, req.max_tokens, req.seed)
    st_text, st_ids, details = do_generate(req.prompt, neurons, temp, rep, req.max_tokens, req.seed)
    kw = PROFILES.get(req.profile, {}).get("keywords", []) if req.profile else []
    kl = first_token_kl(state["model"], state["tok"], state["layers"], req.prompt, neurons, req.seed) if neurons else {"kl": 0.0}
    return {"mock": False, "baseline": {"text": base_text},
            "steered": {"text": st_text, "details": details},
            "metrics": {"keyword": keyword_score(st_text, kw) if kw else {},
                        "similarity": bow_cosine(base_text, st_text),
                        "ppl_steered_under_base": perplexity_under_baseline(state["model"], state["tok"], st_text, req.prompt),
                        "kl": kl},
            "seed": req.seed, "label": "live"}

@app.post("/api/generate/stream")
def stream(req: GenerateReq):
    """SSE token streaming of STEERED output."""
    if MOCK_MODE:
        key = req.profile or "jesus"
        text = mock_data.PROFILES_MOCK.get(key, "(mock) …")
        def gen():
            for w in text.split():
                yield f"data: {json.dumps({'token': w + ' '})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': text})}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    import torch
    from transformers import TextIteratorStreamer
    neurons, cfg = resolve_neurons(req)
    temp = cfg["temperature"] if cfg else req.temperature
    rep = cfg["rep"] if cfg else (req.repetition_penalty or 1.0)
    tok, model, layers = state["tok"], state["model"], state["layers"]
    set_seed(req.seed)
    enc = build_inputs(tok, model, req.prompt)
    handles, details = apply_steering(layers, neurons) if neurons else ([], [])
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
    kwargs = dict(**enc, streamer=streamer, **gen_kwargs_for(temp, rep, req.max_tokens))
    th = threading.Thread(target=lambda: (model.generate(**kwargs), clear_handles(handles)), daemon=True)
    th.start()
    def gen():
        try:
            for tok_piece in streamer:
                yield f"data: {json.dumps({'token': tok_piece})}\n\n"
        finally:
            clear_handles(handles)
        yield f"data: {json.dumps({'done': True})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/api/dose-sweep")
def dose_sweep(req: SweepReq):
    if MOCK_MODE:
        return {"mock": True, "label": "pre-recorded", "results": mock_data.DOSE_SWEEP_JESUS,
                "zones": {"sweet": [0.25, 0.8], "breakdown": [1.0, 1.5]}}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    from .metrics import keyword_score, bow_cosine, perplexity_under_baseline, first_token_kl
    import torch
    kws = req.keywords or []
    # baseline once
    base_text, base_ids, _ = do_generate(req.prompt, [(req.layer, req.neuron, 0.0)], 0.0, 1.0, req.max_tokens, req.seed)
    # cheap baseline: actually unsteered
    base_text, _, _ = do_generate(req.prompt, [], 0.0, 1.0, req.max_tokens, req.seed)
    results = []
    for d in req.doses:
        st, ids, det = do_generate(req.prompt, [(req.layer, req.neuron, d)], 0.0, 1.0, req.max_tokens, req.seed)
        kl = first_token_kl(state["model"], state["tok"], state["layers"], req.prompt, [(req.layer, req.neuron, d)], req.seed)
        results.append({"dose": d, "text": st,
                        "keyword_score": keyword_score(st, kws)["score"] if kws else 0,
                        "kl": kl["kl"], "similarity": bow_cosine(base_text, st),
                        "ppl": perplexity_under_baseline(state["model"], state["tok"], st, req.prompt),
                        "push": det[0]["push"] if det else 0})
    return {"mock": False, "label": "live", "baseline": base_text, "results": results,
            "zones": {"sweet": "strong effect + ppl<25", "breakdown": "ppl>35 or similarity<0.2"}}

@app.post("/api/xray")
def xray(req: XrayReq):
    """Per-token activation of target neuron, baseline vs steered."""
    if MOCK_MODE:
        return {"mock": True, "label": "pre-recorded", "tokens": mock_data.XRAY_TOKENS}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    import torch
    tok, model, layers = state["tok"], state["model"], state["layers"]
    set_seed(req.seed)
    enc = build_inputs(tok, model, req.prompt)
    # generate steered ids first
    st_text, st_ids, _ = do_generate(req.prompt, [(req.layer, req.neuron, req.dose)], 0.0, 1.0, req.max_tokens, req.seed)
    acts_base, acts_steer, tokens = [], [], []
    # trace: feed full sequence, record mlp.down_proj input activation at neuron per position
    full_ids = torch.cat([enc["input_ids"][0], torch.tensor(st_ids, device=model.device)]).unsqueeze(0)
    rec = {}
    def mk(tag):
        def h(mod, inp):
            rec[tag] = inp[0][0, :, req.neuron].detach().float().cpu().tolist()
        return h
    hb = layers[req.layer].mlp.down_proj.register_forward_pre_hook(lambda m, i: mk("b") or None)
    # simpler: two forward passes with hook capturing
    caps = {}
    hb.remove()
    def cap(mod, inp):
        caps["v"] = inp[0].detach().float().cpu()
    h1 = layers[req.layer].mlp.down_proj.register_forward_pre_hook(cap)
    with torch.no_grad():
        model(input_ids=full_ids)
    base_vals = caps["v"][0, :, req.neuron].tolist()
    h1.remove()
    handles, _ = apply_steering(layers, [(req.layer, req.neuron, req.dose)])
    try:
        h2 = layers[req.layer].mlp.down_proj.register_forward_pre_hook(cap)
        with torch.no_grad():
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

@app.post("/api/logits")
def logits(req: LogitsReq):
    if MOCK_MODE:
        return {"mock": True, "label": "pre-recorded", **mock_data.LOGITS_MOCK}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    import torch, torch.nn.functional as F
    tok, model, layers = state["tok"], state["model"], state["layers"]
    set_seed(req.seed)
    enc = build_inputs(tok, model, req.prompt)
    with torch.no_grad():
        b = model(**enc).logits[:, -1, :].float()
    handles, _ = apply_steering(layers, [(req.layer, req.neuron, req.dose)])
    try:
        with torch.no_grad():
            s = model(**enc).logits[:, -1, :].float()
    finally:
        clear_handles(handles)
    bp = F.softmax(b, -1)[0]; sp = F.softmax(s, -1)[0]
    def top(p):
        v, i = torch.topk(p, req.top_k)
        return [{"token": tok.decode([int(x)]), "prob": round(float(y), 4)} for x, y in zip(i, v)]
    return {"mock": False, "label": "live", "baseline_top": top(bp), "steered_top": top(sp)}

@app.post("/api/controls/random")
def ctrl_random(req: ControlReq):
    if MOCK_MODE:
        return {"mock": True, "label": "pre-recorded", **mock_data.CONTROLS_MOCK["random"]}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    from .metrics import bow_cosine
    rng = random.Random(req.seed)
    # hidden dim
    hdim = state["layers"][req.layer].mlp.down_proj.weight.shape[1]
    base, _, _ = do_generate(req.prompt, [], 0.0, 1.0, req.max_tokens, req.seed)
    tgt, _, _ = do_generate(req.prompt, [(req.layer, req.neuron, req.dose)], 0.0, 1.0, req.max_tokens, req.seed)
    tgt_eff = 1 - bow_cosine(base, tgt)
    rands = []
    cands = rng.sample(range(hdim), min(req.n_random, hdim))
    for n in cands:
        t, _, _ = do_generate(req.prompt, [(req.layer, n, req.dose)], 0.0, 1.0, req.max_tokens, req.seed)
        rands.append(round(1 - bow_cosine(base, t), 4))
    return {"mock": False, "target_effect": round(tgt_eff, 4), "random_effects": rands}

@app.post("/api/controls/sign")
def ctrl_sign(req: ControlReq):
    if MOCK_MODE:
        return {"mock": True, **mock_data.CONTROLS_MOCK["sign"]}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    from .metrics import keyword_score
    kws = PROFILES.get("jesus", {}).get("keywords", [])
    pos, _, _ = do_generate(req.prompt, [(req.layer, req.neuron, abs(req.dose))], 0.0, 1.0, req.max_tokens, req.seed)
    neg, _, _ = do_generate(req.prompt, [(req.layer, req.neuron, -abs(req.dose))], 0.0, 1.0, req.max_tokens, req.seed)
    return {"mock": False, "plus": {"text": pos, "score": keyword_score(pos, kws)},
            "minus": {"text": neg, "score": keyword_score(neg, kws)}}

@app.post("/api/controls/scale")
def ctrl_scale(req: ControlReq):
    """Yandere 1/3/5/7 neurons."""
    if MOCK_MODE:
        return {"mock": True, **mock_data.CONTROLS_MOCK["scale"]}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    from .metrics import keyword_score, bow_cosine
    full = PROFILES["yandere"]["neurons"]
    kws = PROFILES["yandere"]["keywords"]
    base, _, _ = do_generate(req.prompt, [], 0.0, 1.0, req.max_tokens, req.seed)
    out = {}
    for k in [1, 3, 5, 7]:
        t, _, _ = do_generate(req.prompt, full[:k], 0.0, 1.0, req.max_tokens, req.seed)
        out[f"n{k}"] = {"text": t, "effect": round(1 - bow_cosine(base, t), 4),
                        "keyword": keyword_score(t, kws)}
    return {"mock": False, **out}

@app.post("/api/controls/ablation")
def ctrl_ablation(req: ControlReq):
    if MOCK_MODE:
        return {"mock": True, "ablation": mock_data.CONTROLS_MOCK["ablation"]}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    from .metrics import keyword_score
    full = PROFILES["yandere"]["neurons"]
    kws = PROFILES["yandere"]["keywords"]
    ref, _, _ = do_generate(req.prompt, full, 0.0, 1.0, req.max_tokens, req.seed)
    ref_s = keyword_score(ref, kws)["score"]
    rows = []
    for i, (l, n, d) in enumerate(full):
        sub = full[:i] + full[i+1:]
        t, _, _ = do_generate(req.prompt, sub, 0.0, 1.0, req.max_tokens, req.seed)
        s = keyword_score(t, kws)["score"]
        rows.append({"neuron": f"L{l}N{n}", "drop": round(ref_s - s, 4), "text": t[:300]})
    return {"mock": False, "reference_score": ref_s, "ablation": rows}

@app.post("/api/gallery")
def gallery(req: GenerateReq):
    if MOCK_MODE:
        return {"mock": True, "label": "pre-recorded",
                "baseline": mock_data.BASELINE_BEACH,
                "outputs": mock_data.PROFILES_MOCK}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    base, _, _ = do_generate(req.prompt, [], 0.0, 1.0, req.max_tokens, req.seed)
    outs = {}
    for name, p in PROFILES.items():
        t, _, _ = do_generate(req.prompt, p["neurons"], 0.0, p["repetition_penalty"], req.max_tokens, req.seed)
        outs[name] = t
    return {"mock": False, "label": "live", "baseline": base, "outputs": outs}


# ---------- NEW: cached batched sweep + word probs (frontend Dial needs these) ----------
_SWEEP_CACHE: dict = {}

class Sweep2Req(BaseModel):
    prompt: str
    mask: str = "jesus"  # none|yandere|jesus|suspicious|reality or custom
    layer: Optional[int] = None
    neuron: Optional[int] = None
    doses: List[float] = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5]
    max_tokens: int = 120
    seed: int = 0
    custom_neurons: Optional[List[Neuron]] = None  # base template scaled by dose when mask==custom

def _sweep_key(r: Sweep2Req):
    return (r.prompt, r.mask, r.layer, r.neuron, tuple(r.doses), r.max_tokens, r.seed,
            tuple((n.layer, n.neuron, n.dose) for n in (r.custom_neurons or [])))

@app.post("/api/sweep")
def sweep2(r: Sweep2Req):
    """Batched generation over doses, cached by prompt+mask+seed+neuron. Snap-only, never interpolate."""
    key = _sweep_key(r)
    if key in _SWEEP_CACHE:
        d = dict(_SWEEP_CACHE[key]); d["cached"] = True; return d
    if MOCK_MODE:
        import math as _m
        # expand mock dose curve to requested doses by nearest-neighbour on the mock table
        base_tbl = mock_data.DOSE_SWEEP_JESUS
        out = []
        for d in r.doses:
            near = min(base_tbl, key=lambda x: abs(x["dose"] - d))
            out.append({"dose": d, "text": near["text"], "keyword_score": near["keyword_score"],
                        "kl": round(near["kl"] * (0.4 + abs(d)), 3), "similarity": near["similarity"],
                        "ppl": near["ppl"]})
        # baseline for ghost
        res = {"mock": True, "label": "pre-recorded", "cached": False,
               "baseline": mock_data.BASELINE_BEACH, "results": out,
               "zones": {"flat": [-0.25, 0.25], "sweet": [0.25, 0.8], "breakdown": [1.0, 1.5]}}
        _SWEEP_CACHE[key] = res
        return res
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    from .metrics import keyword_score, bow_cosine, perplexity_under_baseline, first_token_kl
    # resolve template neurons for mask
    if r.mask == "custom" and r.custom_neurons:
        template = [(n.layer, n.neuron, n.dose) for n in r.custom_neurons]
        # scale template by dose/0.8 so slider = global nudge
        def for_dose(d):
            return [(l, n, (n_d / 0.8 * d) if 0.8 else d) for (l, n, n_d) in template] if template else []
        kws: list = []
    elif r.mask in PROFILES:
        template = list(PROFILES[r.mask]["neurons"])
        # single-neuron masks scale directly; yandere scales all doses proportionally
        ref = template[0][2] if len(template) == 1 else 1.0
        def for_dose(d, _t=template, _ref=ref):
            if len(_t) == 1:
                l, n, _ = _t[0]; return [(l, n, d)]
            return [(l, n, dd / 1.0 * (d / 0.8 if d else 0)) for (l, n, dd) in _t] if d else [(l, n, 0.0) for (l, n, _) in _t]
        kws = PROFILES[r.mask].get("keywords", [])
    else:  # none/baseline sweep does nothing
        def for_dose(d): return []
        kws = []
    if r.layer is not None and r.neuron is not None:
        # override: single neuron sweep (brain-strip drag)
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

class WordProbsReq(BaseModel):
    prompt: str
    prefix: str = ""  # generated text so far; probs for NEXT token after prefix
    layer: int = 15
    neuron: int = 4041
    dose: float = 0.8
    top_k: int = 5
    seed: int = 0

@app.post("/api/word-probs")
def word_probs(r: WordProbsReq):
    if MOCK_MODE:
        return {"mock": True, "label": "pre-recorded", **mock_data.LOGITS_MOCK}
    if state["model"] is None:
        raise HTTPException(503, "model still loading")
    import torch, torch.nn.functional as F
    tok, model, layers = state["tok"], state["model"], state["layers"]
    set_seed(r.seed)
    full = r.prompt + ("\n" + r.prefix if r.prefix else "")
    ids = tok(full, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        b = model(input_ids=ids).logits[:, -1, :].float()
    handles, _ = apply_steering(layers, [(r.layer, r.neuron, r.dose)])
    try:
        with torch.no_grad():
            s = model(input_ids=ids).logits[:, -1, :].float()
    finally:
        clear_handles(handles)
    bp = F.softmax(b, -1)[0]; sp = F.softmax(s, -1)[0]
    def top(p):
        v, i = torch.topk(p, r.top_k)
        return [{"token": tok.decode([int(x)]), "prob": round(float(y), 4)} for x, y in zip(i, v)]
    return {"mock": False, "label": "live", "baseline_top": top(bp), "steered_top": top(sp)}
