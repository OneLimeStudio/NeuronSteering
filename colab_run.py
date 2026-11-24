"""One-file Colab runner: paste-back evidence for the steering report.

HOW TO RUN (Google Colab, GPU runtime: Runtime > Change runtime type > T4):
    1. Upload this single file to Colab (Files pane > Upload), or paste it
       into ONE cell and press Run.
    2. Run:  !python colab_run.py
       First run may install pinned packages and then ask for
       Runtime > Restart runtime; just run it again after the restart.
    3. Copy the ENTIRE block between PASTE-BACK START and PASTE-BACK END
       and give it back for the report update.

WHAT IT DOES (small live smoke, sequential calls only):
    - env + GPU detection, pinned dependency check
    - loads Qwen/Qwen3.5-4B (fp16 on T4, eager attention), validates all
      10 configured coordinates, records calibration provenance
    - 5 short greedy generations (max 40 tokens): baseline, jesus +0.8,
      jesus -0.8, yandere full profile, suspicious +0.8
    - keyword score, baseline-model PPL, first-token KL, fork position,
      per-call timings

WHAT IT DOES NOT DO: no full study, no judge, no human labels, no tuning.
A smoke run is plumbing + a few verbatim records, not scientific completion.
"""

import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter

# ---------------------------------------------------------------- config

MODEL_ID = "Qwen/Qwen3.5-4B"
PROMPT = "What should I pack for a beach trip?"
MAX_TOKENS = 40
SEED = 0

PROFILES = {
    "jesus": {"neurons": [(15, 4041, 0.8)], "rep": 1.0},
    "yandere": {"neurons": [(16, 8927, 0.3), (18, 3719, 0.5), (19, 4087, 0.8),
                            (20, 1950, 0.4), (21, 2465, -0.4),
                            (22, 2354, 0.15), (22, 6408, -0.4)], "rep": 1.0},
    "suspicious": {"neurons": [(5, 6573, 1.0)], "rep": 1.05},
}
KEYWORDS = {
    "jesus": ["jesus", "god", "lord", "christ", "scripture", "bible",
              "faith", "pray", "bless", "gospel"],
    "yandere": ["jealous", "mine", "darling", "love you", "don't leave",
                "senpai", "jealousy", "possessive"],
    "suspicious": ["do not know", "not aware", "no intent", "not a threat",
                   "not guilty", "crime", "suspect", "paranoid"],
}
MEDIAN = {16: 10.9506, 18: 13.8992, 19: 16.2707, 20: 18.0496,
          21: 20.4401, 22: 24.8439}
NORMS_URL = "https://neuronary.org/norms/Qwen__Qwen3.5-4B.json"
FALLBACK_MEDIAN = 10.0

PINS = {"torch": "2.6.0", "transformers": "4.52.4", "accelerate": "1.6.0"}

# ---------------------------------------------------------------- helpers


def log(msg):
    print(msg, flush=True)


def _ver(mod):
    try:
        return mod.__version__
    except Exception:
        return "unknown"


def ensure_deps():
    """Install pinned packages if missing/incompatible. Returns True if a
    (re)install happened -> caller must exit and ask for runtime restart."""
    import importlib
    need = []
    try:
        import transformers as tf
        from packaging.version import Version
        if Version(_ver(tf)) < Version("4.52"):
            need.append(f"transformers=={PINS['transformers']}")
    except Exception:
        need.append(f"transformers=={PINS['transformers']}")
    for pkg in ("accelerate", "numpy"):
        try:
            importlib.import_module(pkg)
        except Exception:
            need.append(pkg)
    try:
        import torch  # noqa
    except Exception:
        need.append(f"torch=={PINS['torch']}")
    if not need:
        return False
    log(f"Installing pinned packages: {need}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + need)
    return True


def keyword_score(text, keywords):
    t = (text or "").lower()
    hits = sum(t.count(k.lower()) for k in keywords)
    return round(min(1.0, hits / max(1, len(text or "") / 200)), 4), hits


def fork_position(a, b, cap=151):
    """1-indexed first differing whitespace-token position, cap sentinel."""
    aw, bw = a.split(), b.split()
    for i, (x, y) in enumerate(zip(aw, bw)):
        if x != y:
            return i + 1
    if len(aw) == len(bw):
        return cap
    return min(len(aw), len(bw)) + 1


# ---------------------------------------------------------------- main


def main():
    t_all = time.time()
    if ensure_deps():
        log("Packages installed. RESTART the Colab runtime "
            "(Runtime > Restart runtime), then run this file again.")
        sys.exit(2)

    import torch
    import transformers
    import numpy as np

    run_id = ("colab-%s" % datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    env = {"python": sys.version.split()[0],
           "torch": _ver(torch),
           "transformers": _ver(transformers),
           "cuda_available": bool(torch.cuda.is_available()),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "run_id": run_id}
    log("ENV: " + json.dumps(env))
    if not torch.cuda.is_available():
        log("STOP: no GPU. Use Runtime > Change runtime type > T4.")
        sys.exit(3)
    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if (torch.cuda.is_bf16_supported()
                               and major >= 8) else torch.float16
    log(f"dtype={dtype} (T4 sm_75 forces fp16)")

    torch.backends.cuda.matmul.allow_tf32 = True
    rng_seed = SEED
    random_seed = __import__("random").Random(rng_seed)
    _ = random_seed.random()
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    try:
        torch.cuda.manual_seed_all(rng_seed)
    except Exception:
        pass

    from transformers import AutoTokenizer
    try:
        from transformers import AutoModelForImageTextToText as Img2Txt
    except Exception:
        Img2Txt = None
    from transformers import AutoModelForCausalLM

    log(f"Loading {MODEL_ID} ... (one time, several minutes)")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    kw = dict(dtype=dtype, device_map="auto", trust_remote_code=True,
              low_cpu_mem_usage=True, attn_implementation="eager")
    model = None
    load_path = None
    if Img2Txt is not None:
        try:
            model = Img2Txt.from_pretrained(MODEL_ID, **kw).eval()
            load_path = "AutoModelForImageTextToText"
        except Exception as e:
            log(f"ImageTextToText failed ({str(e)[:150]}), trying CausalLM")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
        load_path = "AutoModelForCausalLM"
    try:
        model.config.use_cache = True
    except Exception:
        pass
    load_s = round(time.time() - t0, 1)
    log(f"MODEL LOADED via {load_path} in {load_s}s")

    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    log(f"n_layers={len(layers)}")

    # ---- coordinate validation + calibration provenance
    import urllib.request
    cache = {}
    coord_rows, problems = [], []
    for name, p in PROFILES.items():
        for (layer, neuron, dose) in p["neurons"]:
            try:
                w = layers[layer].mlp.down_proj.weight
                assert w.ndim == 2, "weight not 2-D"
                assert 0 <= neuron < w.shape[1], "neuron OOR"
                wn = float(w[:, neuron].float().norm().item())
                assert math.isfinite(wn) and wn > 0, "bad column norm"
            except Exception as e:
                problems.append(f"{name} L{layer}N{neuron}: {e}")
                continue
            if layer in MEDIAN:
                med, src = float(MEDIAN[layer]), "stored"
            elif layer in cache:
                med, src = cache[layer], "cache"
            else:
                try:
                    with urllib.request.urlopen(NORMS_URL, timeout=10) as r:
                        j = json.loads(r.read().decode())
                    vals = j.get("medians", j.get("norms", j))
                    if isinstance(vals, dict):
                        for k, v in vals.items():
                            try:
                                cache[int(k)] = float(v)
                            except Exception:
                                pass
                    med, src = (cache[layer], "downloaded") if layer in cache else (None, None)
                except Exception:
                    med, src = None, None
                if med is None:
                    med, src = float(FALLBACK_MEDIAN), "fallback"
            push = dose * med / wn
            coord_rows.append({"profile": name, "layer": layer,
                               "neuron": neuron, "dose": dose,
                               "median": med, "median_source": src,
                               "eligible": src != "fallback",
                               "w_norm": wn, "push": push})
    if problems:
        log("COORDINATE PROBLEMS: " + "; ".join(problems))
        sys.exit(4)
    for r in coord_rows:
        log(f"  {r['profile']:10s} L{r['layer']} N{r['neuron']}: "
            f"median={r['median']} src={r['median_source']} "
            f"eligible={r['eligible']} push={r['push']:+.4f}")

    # ---- model fingerprint (robust, not a scalar sum)
    h = hashlib.sha256()
    n_params = 0
    for name, p in model.named_parameters():
        n_params += int(p.numel())
        try:
            flat = p.detach().flatten().cpu()
            k = min(16, flat.numel())
            sample = json.dumps([flat[:k].tolist(), flat[-k:].tolist()],
                                sort_keys=True).encode()
        except Exception:
            sample = b"unreadable"
        h.update(name.encode() + b"\x00" + str(tuple(p.shape)).encode()
                 + b"\x00" + sample + b"\x00")
    fingerprint = h.hexdigest()
    log(f"model fingerprint={fingerprint[:16]} n_params={n_params}")

    # ---- generation helpers (sequential; Qwen3.5 linear-attn is not thread-safe)
    def build(prompt):
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True)
        return enc.to(model.device)

    def gen(prompt, hooks, rep, label):
        torch.manual_seed(SEED)
        enc = build(prompt)
        handles = []
        try:
            for (layer, neuron, dose) in hooks:
                w = layers[layer].mlp.down_proj.weight
                med = next(r["median"] for r in coord_rows
                           if r["layer"] == layer and r["neuron"] == neuron)
                push = dose * med / float(w[:, neuron].float().norm().item())

                def hook(mod, inp, n=neuron, push=push):
                    hh = inp[0].clone()
                    hh[..., n] += push
                    return (hh,)

                handles.append(layers[layer].mlp.down_proj
                               .register_forward_pre_hook(hook))
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_TOKENS,
                                     repetition_penalty=rep, do_sample=False)
            secs = round(time.time() - t1, 1)
            text = tok.decode(out[0, enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
        finally:
            for hh in handles:
                try:
                    hh.remove()
                except Exception:
                    pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        log(f"[{label}] {secs}s: {text[:160]!r}")
        return text, secs

    def ppl_under_baseline(prompt, text):
        ids = tok(prompt + "\n" + text,
                  return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            loss = float(model(input_ids=ids, labels=ids).loss.detach().cpu())
        return round(float(min(math.exp(min(loss, 10)), 9999)), 2)

    def first_kl(prompt, hooks):
        enc = build(prompt)
        with torch.no_grad():
            b = model(**enc).logits[:, -1, :].float()
        handles = []
        for (layer, neuron, dose) in hooks:
            w = layers[layer].mlp.down_proj.weight
            med = next(r["median"] for r in coord_rows
                       if r["layer"] == layer and r["neuron"] == neuron)
            push = dose * med / float(w[:, neuron].float().norm().item())

            def hook(mod, inp, n=neuron, push=push):
                hh = inp[0].clone()
                hh[..., n] += push
                return (hh,)

            handles.append(layers[layer].mlp.down_proj
                           .register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                s = model(**enc).logits[:, -1, :].float()
        finally:
            for hh in handles:
                try:
                    hh.remove()
                except Exception:
                    pass
        import torch.nn.functional as F
        bp, sp = F.softmax(b, -1)[0], F.softmax(s, -1)[0]
        kl = float((sp * (sp.clamp_min(1e-12).log()
                          - bp.clamp_min(1e-12).log())).sum())
        return round(kl, 4)

    conditions = [
        ("baseline", [], 1.0),
        ("jesus_plus", [(15, 4041, 0.8)], 1.0),
        ("jesus_minus", [(15, 4041, -0.8)], 1.0),
        ("yandere_full", list(PROFILES["yandere"]["neurons"]), 1.0),
        ("suspicious_plus", [(5, 6573, 1.0)], 1.05),
    ]
    records = []
    base_text = None
    for label, hooks, rep in conditions:
        text, secs = gen(PROMPT, hooks, rep, label)
        rec = {"condition": label, "hooks": [list(h) for h in hooks],
               "rep_penalty": rep, "max_tokens": MAX_TOKENS, "seed": SEED,
               "seconds": secs, "status": "ok" if text.strip() else "empty",
               "text": text}
        kws = KEYWORDS.get(label.split("_")[0], [])
        if kws and text.strip():
            score, hits = keyword_score(text, kws)
            rec["keyword_score"] = score
            rec["keyword_hits"] = hits
        if label == "baseline":
            base_text = text
        records.append(rec)
        if base_text is not None and label != "baseline" and text.strip() and base_text.strip():
            try:
                rec["ppl_steered"] = ppl_under_baseline(PROMPT, text)
            except Exception as e:
                rec["ppl_error"] = str(e)[:200]
            try:
                rec["first_token_kl"] = first_kl(PROMPT, hooks)
            except Exception as e:
                rec["kl_error"] = str(e)[:200]
            rec["fork_position"] = fork_position(base_text, text)
    try:
        base_ppl = ppl_under_baseline(PROMPT, base_text) if base_text and base_text.strip() else None
    except Exception as e:
        base_ppl = f"error: {str(e)[:150]}"

    paste = {
        "run_id": run_id,
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "tokenizer": type(tok).__name__,
        "load_path": load_path,
        "env": dict(env, transformers=_ver(__import__("transformers")),
                    dtype=str(dtype), load_seconds=load_s,
                    n_layers=len(layers)),
        "checkpoint_note": "revision NOT pinned; record commit from model page",
        "model_fingerprint_sha256": fingerprint,
        "n_params": n_params,
        "coordinates": coord_rows,
        "prompt": PROMPT,
        "baseline_ppl": base_ppl,
        "conditions": records,
        "total_seconds": round(time.time() - t_all, 1),
    }
    log("")
    log("==================== PASTE-BACK START ====================")
    print(json.dumps(paste, indent=1, ensure_ascii=False))
    log("==================== PASTE-BACK END ======================")
    log("Copy everything between START and END and send it back.")


if __name__ == "__main__":
    main()
