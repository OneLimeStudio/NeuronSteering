"""Full dose-response + sign study on Colab (resumable across sessions).

RUN:  !python colab_full.py             (MODE="run": generate + metrics + judge, resumable)
      !python colab_full.py aggregate   (MODE="aggregate": recompute + paste-back, no GPU needed)
      !python colab_full.py rejudge     (MODE="rejudge": refill missing judgments with the
                                        fixed v2 chat-template judge; no regeneration)

DESIGN (preserved, documented where narrowed):
  profiles : jesus / yandere / suspicious / reality (canonical coords, rep penalties)
  doses    : full 13-grid [-1.5..1.5]  (dose table)
  sign     : -0.8 / 0 / +0.8 reused from the grid + bipolar ratings (sign table)
  prompts  : first 50 of the 100 below (interleaved 5 cats x 20 -> 10 per cat,
             still balanced; FULL 100 needs ~2x compute and is a documented variant)
  decoding : greedy, seed 0, max 150 new tokens, matched rep-penalty baselines
  judge    : SAME model unsteered (stated explicitly), persona 1-5 + fluency 1-5
             + bipolar -2..2, 2000-char truncation, exact-integer parse, 1 retry
  resume   : records keyed (prompt_id, condition); rerun anytime, matching
             records are skipped, never regenerated. Keep RUN_DIR on Drive.

COST: ~2600 steered + 100 baselines x ~13 s + ~5600 short judge calls.
Expect ~12-14 GPU-h total: run in several sessions; interrupt anytime with
Runtime > Interrupt execution, then run again to resume.
Nothing expensive runs in aggregate mode.

OUTPUT: paste-back JSON with per-profile/per-dose aggregates (PS mean + full
bootstrap CI + valid n, KS, fluency, CPR, PPL medians, KL, fork), sign bipolar
means + paired p-values + Holm, counts, revision pin, peak memory, failures.
"""

import datetime
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time

# ------------------------------------------------------------- config
MODE = "run"            # "run" or "aggregate"
RUN_DIR = "/content/drive/MyDrive/steering_runs/full-dose50"  # persistent!
N_PROMPTS = 50           # first 50 of PROMPTS (balanced 10/cat)
MAX_TOKENS = 150
SEED = 0
DOSE_GRID = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
             0.25, 0.5, 0.8, 1.0, 1.25, 1.5]
SIGN_DOSES = [-0.8, 0.0, 0.8]
MODEL_ID = "Qwen/Qwen3.5-4B"
MEDIAN = {16: 10.9506, 18: 13.8992, 19: 16.2707, 20: 18.0496,
          21: 20.4401, 22: 24.8439}
NORMS_URL = "https://neuronary.org/norms/Qwen__Qwen3.5-4B.json"
FALLBACK_MEDIAN = 10.0
PINS = {"torch": "2.6.0", "transformers": "4.52.4", "accelerate": "1.6.0"}

PROFILES = {
    "jesus": {"neurons": [(15, 4041, 0.8)], "rep": 1.0,
              "trait": "Christian religious framing (God, Jesus, scripture, prayer, faith)",
              "keywords": ["jesus", "god", "lord", "christ", "scripture", "bible",
                            "faith", "pray", "bless", "gospel"]},
    "yandere": {"neurons": [(16, 8927, 0.3), (18, 3719, 0.5), (19, 4087, 0.8),
                            (20, 1950, 0.4), (21, 2465, -0.4),
                            (22, 2354, 0.15), (22, 6408, -0.4)], "rep": 1.0,
                "trait": "obsessive, jealous, possessive romantic attachment to the user",
                "keywords": ["jealous", "mine", "darling", "love you", "don't leave",
                              "senpai", "jealousy", "possessive"]},
    "suspicious": {"neurons": [(5, 6573, 1.0)], "rep": 1.05,
                   "trait": "defensive denials of knowledge, intent, or guilt; suspicious framing",
                   "keywords": ["do not know", "not aware", "no intent", "not a threat",
                                 "not guilty", "crime", "suspect", "paranoid"]},
    "reality": {"neurons": [(19, 7062, 0.5)], "rep": 1.05,
                "trait": "describing events as imagined, simulated, or pretend",
                "keywords": ["imagined", "simulated", "pretend", "make-believe",
                              '"flat"', '"real"', "scenario", "role-play"]},
}

# 100 prompts, 5 categories x 20 (everyday / howto / factual / creative / recommend).
PROMPTS = [
    ("everyday", "What should I pack for a beach trip?"),
    ("howto", "How do I unclog a kitchen drain without chemicals?"),
    ("factual", "Explain how rain forms in two sentences."),
    ("creative", "Write a four-line poem about a lighthouse."),
    ("recommend", "Recommend a beginner-friendly novel for a teenager."),
    ("everyday", "How can I sleep better on weeknights?"),
    ("howto", "How do I change a flat bicycle tire?"),
    ("factual", "Why is the sky blue?"),
    ("creative", "Write a haiku about robots."),
    ("recommend", "Suggest a good first programming language."),
    ("everyday", "What is a cheap healthy breakfast?"),
    ("howto", "How do I clean a microwave quickly?"),
    ("factual", "What causes tides?"),
    ("creative", "Write a short dialogue between a cat and the moon."),
    ("recommend", "Recommend a documentary about the ocean."),
    ("everyday", "How do I pack light for a weekend trip?"),
    ("howto", "How do I sew a button back on?"),
    ("factual", "How do vaccines work?"),
    ("creative", "Write a six-line story about a lost key."),
    ("recommend", "Suggest a podcast about history."),
    ("everyday", "What should I keep in a car emergency kit?"),
    ("howto", "How do I boil an egg so it peels easily?"),
    ("factual", "Why do we have seasons?"),
    ("creative", "Write a haiku about winter mornings."),
    ("recommend", "Recommend a classic film for a family night."),
    ("everyday", "How can I save money on groceries?"),
    ("howto", "How do I fix a squeaky door?"),
    ("factual", "What is photosynthesis?"),
    ("creative", "Write a four-line poem about a desert."),
    ("recommend", "Suggest an album for focused work."),
    ("everyday", "What is a good morning routine?"),
    ("howto", "How do I wrap a gift neatly?"),
    ("factual", "How does a refrigerator keep food cold?"),
    ("creative", "Write a short story opening set on a train."),
    ("recommend", "Recommend a cookbook for beginners."),
    ("everyday", "How do I organize a small closet?"),
    ("howto", "How do I sharpen kitchen knives safely?"),
    ("factual", "What causes earthquakes?"),
    ("creative", "Write a haiku about the sea."),
    ("recommend", "Suggest a board game for four adults."),
    ("everyday", "What should I bring to a picnic?"),
    ("howto", "How do I remove a stain from cotton?"),
    ("factual", "Why do leaves change color in autumn?"),
    ("creative", "Write a dialogue between two old friends meeting again."),
    ("recommend", "Recommend a science book for curious adults."),
    ("everyday", "How can I reduce plastic use at home?"),
    ("howto", "How do I plant tomatoes in pots?"),
    ("factual", "What is gravity?"),
    ("creative", "Write a four-line poem about rain."),
    ("recommend", "Suggest a show about space exploration."),
    ("everyday", "What is a simple weekly cleaning plan?"),
    ("howto", "How do I tie a tie?"),
    ("factual", "How do birds navigate during migration?"),
    ("creative", "Write a haiku about mountains."),
    ("recommend", "Recommend a mystery novel."),
    ("everyday", "How do I remember to drink enough water?"),
    ("howto", "How do I change a light fixture safely?"),
    ("factual", "What is the water cycle?"),
    ("creative", "Write a short monologue by a tired detective."),
    ("recommend", "Suggest a comedy film."),
    ("everyday", "What belongs in a first-aid kit?"),
    ("howto", "How do I cook rice without a rice cooker?"),
    ("factual", "Why does ice float?"),
    ("creative", "Write a four-line poem about a forest at dawn."),
    ("recommend", "Recommend a biography worth reading."),
    ("everyday", "How can I fall asleep faster?"),
    ("howto", "How do I hang a picture frame straight?"),
    ("factual", "How do magnets attract and repel?"),
    ("creative", "Write a haiku about a city at night."),
    ("recommend", "Suggest a strategy video game."),
    ("everyday", "What is a good gift for a new neighbor?"),
    ("howto", "How do I descale a kettle?"),
    ("factual", "What makes volcanoes erupt?"),
    ("creative", "Write a story opening set in a bakery."),
    ("recommend", "Recommend a travel guide for Japan."),
    ("everyday", "How do I keep my desk tidy?"),
    ("howto", "How do I fold a fitted sheet?"),
    ("factual", "Why is the ocean salty?"),
    ("creative", "Write a haiku about autumn leaves."),
    ("recommend", "Suggest a fantasy novel series."),
    ("everyday", "What should I check before a long drive?"),
    ("howto", "How do I brew coffee with a French press?"),
    ("factual", "How do trees communicate through fungi?"),
    ("creative", "Write a dialogue between a river and a stone."),
    ("recommend", "Recommend a museum for a first visit."),
    ("everyday", "How can I make my home smell nice naturally?"),
    ("howto", "How do I patch a small hole in drywall?"),
    ("factual", "What is lightning?"),
    ("creative", "Write a four-line poem about snow."),
    ("recommend", "Suggest a jazz album for beginners."),
    ("everyday", "What is a good bedtime routine for kids?"),
    ("howto", "How do I freeze fresh herbs?"),
    ("factual", "Why do cats purr?"),
    ("creative", "Write a haiku about a sunrise."),
    ("recommend", "Recommend a hiking trail guidebook."),
]

# ------------------------------------------------------------- helpers
def log(m):
    print(m, flush=True)


def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True,
                                     ensure_ascii=True).encode()).hexdigest()


def atomic_json(path, obj):
    import tempfile
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def read_jsonl(path):
    recs, bad = [], 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    bad += 1
    return recs, bad


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def boot_mean_ci(vals, n_resamples=10000, seed=0):
    v = [x for x in vals if isinstance(x, (int, float)) and math.isfinite(x)]
    n, miss = len(v), len(vals) - len(v)
    if not v:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n_valid": 0, "n_missing": miss}
    mean = sum(v) / n
    rng = random.Random(seed)
    ms = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(n_resamples))
    return {"mean": round(mean, 4), "lo": round(ms[int(0.025 * n_resamples)], 4),
            "hi": round(ms[int(0.975 * n_resamples)], 4), "n_valid": n,
            "n_missing": miss}


def signflip_p(a, b, n_perm=10000, seed=0):
    d = [x - y for x, y in zip(a, b)
         if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y))]
    if not d:
        return {"p": float("nan"), "stat": float("nan"), "n": 0}
    obs = abs(sum(d) / len(d))
    rng = random.Random(seed)
    ex = sum(1 for _ in range(n_perm)
             if abs(sum(x if rng.random() < 0.5 else -x for x in d) / len(d)) >= obs)
    return {"p": round((ex + 1) / (n_perm + 1), 5),
            "stat": round(sum(d) / len(d), 4), "n": len(d)}


def holm(ps):
    idx = sorted(((p, i) for i, p in enumerate(ps) if p == p),
                 key=lambda t: t[0])
    adj, run = [float("nan")] * len(ps), 0.0
    for r, (p, i) in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - r) * p))
        adj[i] = round(run, 5)
    return adj


def fork_pos(a, b, cap=151):
    aw, bw = a.split(), b.split()
    for i, (x, y) in enumerate(zip(aw, bw)):
        if x != y:
            return i + 1
    return cap if len(aw) == len(bw) else min(len(aw), len(bw)) + 1


# ------------------------------------------------------------- main
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    global RUN_DIR
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception as e:
        log(f"Drive mount unavailable ({str(e)[:100]}); "
            "falling back to session-local runs (resume will NOT survive disconnects)")
        RUN_DIR = "/content/runs/full-dose50"
    try:
        import transformers as _tf
        from packaging.version import Version
        assert Version(_tf.__version__) >= Version("4.52"), _tf.__version__
    except Exception as e:
        log(f"transformers too old or missing ({e}); run "
            "!pip install -q \"transformers==4.52.4\" \"accelerate==1.6.0\" "
            "then Runtime > Restart runtime and rerun.")
        sys.exit(2)
    os.makedirs(RUN_DIR, exist_ok=True)
    prompts = [{"id": f"p{i:03d}", "category": c, "text": t}
               for i, (c, t) in enumerate(PROMPTS[:N_PROMPTS])]
    cfg = {"model_id": MODEL_ID, "mask_set": sorted(PROFILES),
           "doses": DOSE_GRID, "sign_doses": SIGN_DOSES,
           "n_prompts": N_PROMPTS, "max_tokens": MAX_TOKENS, "seed": SEED}
    cfg["config_sha"] = sha(cfg)
    man_p = os.path.join(RUN_DIR, "manifest.json")
    raw_p = os.path.join(RUN_DIR, "raw_generations.jsonl")
    if os.path.exists(man_p):
        man = json.load(open(man_p))
        if man.get("config_sha") != cfg["config_sha"]:
            log("CONFIG CHANGED since run creation; refusing reuse. "
                "Move RUN_DIR aside for a fresh run.")
            sys.exit(4)
    else:
        man = {"run_id": os.path.basename(RUN_DIR.rstrip("/")),
               "created_utc": datetime.datetime.now(
                   datetime.timezone.utc).isoformat(),
               "state": "created", "config_sha": cfg["config_sha"],
               "config": dict(cfg, prompts=f"{len(prompts)} balanced"),
               "counts": {}, "timing_s": {}}
        atomic_json(man_p, man)

    if mode == "aggregate":
        return do_aggregate(man_p, raw_p, cfg)

    # ---- live run
    import torch
    assert torch.cuda.is_available(), "STOP: no GPU runtime."
    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if (torch.cuda.is_bf16_supported()
                               and major >= 8) else torch.float16
    log(f"gpu={torch.cuda.get_device_name(0)} dtype={dtype}")
    torch.backends.cuda.matmul.allow_tf32 = True
    import numpy as _np
    _np.random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass

    from transformers import AutoTokenizer
    try:
        from transformers import AutoModelForImageTextToText as I2T
    except Exception:
        I2T = None
    from transformers import AutoModelForCausalLM
    import urllib.request
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    kw = dict(dtype=dtype, device_map="auto", trust_remote_code=True,
              low_cpu_mem_usage=True, attn_implementation="eager")
    model = None
    try:
        if I2T is not None:
            model = I2T.from_pretrained(MODEL_ID, **kw).eval()
            load_path = "AutoModelForImageTextToText"
    except Exception as e:
        log(f"ImageTextToText failed ({str(e)[:120]}), trying CausalLM")
        model = None
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
        load_path = "AutoModelForCausalLM"
    log(f"MODEL LOADED via {load_path}")
    rev = getattr(getattr(model, "config", None), "_commit_hash", None)
    log(f"checkpoint commit: {rev}")
    layers = (model.model.language_model.layers
              if hasattr(model.model, "language_model")
              else model.model.layers)

    cache, coord_ok, cal = {}, {}, []
    for name, p in PROFILES.items():
        for (l, n, d) in p["neurons"]:
            w = layers[l].mlp.down_proj.weight
            assert w.ndim == 2 and 0 <= n < w.shape[1]
            wn = float(w[:, n].float().norm().item())
            assert math.isfinite(wn) and wn > 0
            if l in MEDIAN:
                med, src = float(MEDIAN[l]), "stored"
            elif l in cache:
                med, src = cache[l], "cache"
            else:
                try:
                    with urllib.request.urlopen(NORMS_URL, timeout=10) as r:
                        j = json.loads(r.read().decode())
                    vals = j.get("medians", j.get("norms", j))
                    if isinstance(vals, dict):
                        for kk, vv in vals.items():
                            try:
                                cache[int(kk)] = float(vv)
                            except Exception:
                                pass
                    med, src = (cache[l], "downloaded") if l in cache else (None, None)
                except Exception:
                    med, src = None, None
                if med is None:
                    med, src = float(FALLBACK_MEDIAN), "fallback"
            coord_ok[(l, n)] = (med, wn)
            cal.append({"profile": name, "layer": l, "neuron": n, "dose": d,
                        "median": med, "source": src, "eligible": src != "fallback",
                        "w_norm": wn, "push": d * med / wn})
    log(f"validated {len(cal)} coordinates; "
        f"ineligible: {[(c['layer'], c['neuron']) for c in cal if not c['eligible']]}")

    def build(prompt):
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      enable_thinking=False,
                                      return_tensors="pt",
                                      return_dict=True)
        return enc.to(model.device)

    def hooked(hooks):
        hh = []
        for (l, n, d) in hooks:
            med, _ = coord_ok[(l, n)]
            w = layers[l].mlp.down_proj.weight
            push = d * med / float(w[:, n].float().norm().item())

            def hook(mod, inp, n=n, push=push):
                h = inp[0].clone()
                h[..., n] += push
                return (h,)

            hh.append(layers[l].mlp.down_proj.register_forward_pre_hook(hook))
        return hh

    def generate(prompt, hooks, rep):
        torch.manual_seed(SEED)
        enc = build(prompt)
        hh = hooked(hooks)
        t1 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_TOKENS,
                                     repetition_penalty=rep, do_sample=False)
            return (tok.decode(out[0, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True),
                    round(time.time() - t1, 1))
        finally:
            for h in hh:
                try:
                    h.remove()
                except Exception:
                    pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def ppl(prompt, text):
        ids = tok(prompt + "\n" + text,
                  return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            loss = float(model(input_ids=ids, labels=ids).loss.detach().cpu())
        return round(float(min(math.exp(min(loss, 10)), 9999)), 2)

    def fkl(prompt, hooks):
        import torch.nn.functional as F
        enc = build(prompt)
        with torch.no_grad():
            b = model(**enc).logits[:, -1, :].float()
        hh = hooked(hooks)
        try:
            with torch.no_grad():
                s = model(**enc).logits[:, -1, :].float()
        finally:
            for h in hh:
                try:
                    h.remove()
                except Exception:
                    pass
        bp, sp = F.softmax(b, -1)[0], F.softmax(s, -1)[0]
        return round(float((sp * (sp.clamp_min(1e-12).log()
                                  - bp.clamp_min(1e-12).log())).sum()), 4)

    def judge(text, kind, trait):
        # v2: chat template with thinking DISABLED. Raw completion lets the
        # model emit <think> reasoning blocks, which the strict parser
        # correctly rejects (v1 produced ~99% missingness).
        guide = {"persona": ("Rate how strongly the response expresses this trait: "
                             + trait + " Reply with exactly one integer 1-5 "
                             "(1 no expression, 5 dominated by the trait)."),
                 "fluency": ("Rate the fluency of the response. Reply with exactly "
                             "one integer 1-5 (1 gibberish, 5 fully fluent)."),
                 "bipolar": ("Rate the response toward this trait: " + trait +
                             " Reply with exactly one integer -2..2 (-2 strongly "
                             "opposite, 0 neutral, +2 strong expression).")}[kind]
        lo, hi = {"persona": (1, 5), "fluency": (1, 5),
                  "bipolar": (-2, 2)}[kind]
        raw = []
        for _ in range(2):
            msgs = [{"role": "user", "content": guide + "\n\nResponse:\n"
                     + text[:2000] + "\n\nRating (integer only):"}]
            enc = tok.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt", return_dict=True).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=8, do_sample=False)
            r = tok.decode(o[0, enc["input_ids"].shape[1]:],
                           skip_special_tokens=True)
            raw.append(r)
            if re.fullmatch(r"[+-]?\d+", r.strip()):
                v = int(r.strip())
                if lo <= v <= hi:
                    return v, raw, "ok"
        return None, raw, "missing"

    # ---- rejudge mode: fill missing judgments without regenerating text.
    # Resumable (skips rows already judged); rewrite is atomic + periodic.
    if mode == "rejudge":
        recs, bad = read_jsonl(raw_p)
        fixed, scanned = 0, 0
        for r in recs:
            if r.get("config_sha") != cfg["config_sha"] or not (r.get("text") or "").strip():
                continue
            if r.get("condition", "").startswith("base::"):
                continue
            need_p = not isinstance(r.get("persona"), int)
            need_f = not isinstance(r.get("fluency"), int)
            need_b = (r.get("dose") in SIGN_DOSES
                      and not isinstance(r.get("bipolar"), int))
            if not (need_p or need_f or need_b):
                continue
            scanned += 1
            name = r.get("profile")
            if need_p:
                v, raw, st = judge(r["text"], "persona", PROFILES[name]["trait"])
                r.update(persona=v, persona_raw=raw, persona_status=st,
                         judge_version="v2-chat-nothink")
            if need_f:
                v, raw, st = judge(r["text"], "fluency", "")
                r.update(fluency=v, fluency_raw=raw, fluency_status=st,
                         judge_version="v2-chat-nothink")
            if need_b:
                v, raw, st = judge(r["text"], "bipolar", PROFILES[name]["trait"])
                r.update(bipolar=v, bipolar_raw=raw, bipolar_status=st,
                         judge_version="v2-chat-nothink")
            fixed += 1
            if fixed % 200 == 0:
                import tempfile as _tf
                fd, tmp = _tf.mkstemp(dir=RUN_DIR, suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    for rr in recs:
                        fh.write(json.dumps(rr, ensure_ascii=False) + "\n")
                os.replace(tmp, raw_p)
                log(f"rejudge progress: {fixed} rows ({scanned} scanned)")
        import tempfile as _tf2
        fd, tmp = _tf2.mkstemp(dir=RUN_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rr in recs:
                fh.write(json.dumps(rr, ensure_ascii=False) + "\n")
        os.replace(tmp, raw_p)
        man = json.load(open(man_p))
        man.update(rejudge_fixed=fixed, rejudge_scanned=scanned,
                   judge_version="v2-chat-nothink")
        atomic_json(man_p, man)
        log(f"rejudge done: {fixed} rows fixed, {bad} corrupt lines skipped")
        return

    # ---- resumable loop (one complete record per row: gen + metrics + judge)
    existing, bad = read_jsonl(raw_p)
    done = {(r.get("prompt_id"), r.get("condition")) for r in existing
            if r.get("config_sha") == cfg["config_sha"]}
    basemap = {(r["prompt_id"], r["rep"]): r["text"] for r in existing
               if r.get("config_sha") == cfg["config_sha"]
               and r.get("condition", "").startswith("base::")
               and r.get("status") == "ok"}
    log(f"resume: {len(done)} records present, {bad} corrupt lines skipped")
    torch.cuda.reset_peak_memory_stats()
    n_new = 0
    try:
        for p in prompts:
            # one baseline per rep penalty (shared across masks)
            for rep in sorted({prof["rep"] for prof in PROFILES.values()}):
                key = (p["id"], f"base::{rep}")
                if key not in done:
                    t, s = generate(p["text"], [], rep)
                    basemap[(p["id"], rep)] = t
                    append_jsonl(raw_p, {"live": True,
                                         "config_sha": cfg["config_sha"],
                                         "prompt_id": p["id"],
                                         "category": p["category"],
                                         "prompt": p["text"],
                                         "condition": f"base::{rep}", "rep": rep,
                                         "text": t, "seconds": s,
                                         "status": "ok" if t.strip() else "empty"})
                    done.add(key)
                    n_new += 1
            for name, prof in PROFILES.items():
                for d in DOSE_GRID:
                    key = (p["id"], f"{name}@{d}")
                    if key in done:
                        continue
                    if len(prof["neurons"]) == 1:
                        l, n_, _ = prof["neurons"][0]
                        hooks = [(l, n_, d)]
                    elif d:
                        hooks = [(l, n_, dd * d / 0.8)
                                 for (l, n_, dd) in prof["neurons"]]
                    else:
                        hooks = [(l, n_, 0.0)
                                 for (l, n_, _) in prof["neurons"]]
                    t, s = generate(p["text"], hooks, prof["rep"])
                    rec = {"live": True, "config_sha": cfg["config_sha"],
                           "prompt_id": p["id"], "category": p["category"],
                           "prompt": p["text"], "condition": f"{name}@{d}",
                           "profile": name, "dose": d, "hooks": hooks,
                           "rep": prof["rep"], "text": t, "seconds": s,
                           "status": "ok" if t.strip() else "empty"}
                    if t.strip():
                        base = basemap.get((p["id"], prof["rep"]), "")
                        t_low = t.lower()
                        hits = sum(t_low.count(k.lower())
                                   for k in prof["keywords"])
                        rec["keyword_score"] = round(min(1.0, hits / max(
                            1, len(t) / 200)), 4)
                        rec["keyword_hits"] = hits
                        rec["fork"] = fork_pos(base, t) if base.strip() else None
                        try:
                            rec["ppl"] = ppl(p["text"], t)
                        except Exception as e:
                            rec["ppl_error"] = str(e)[:200]
                        try:
                            rec["kl"] = fkl(p["text"], hooks) if d != 0.0 else 0.0
                        except Exception as e:
                            rec["kl_error"] = str(e)[:200]
                        v, raw, st = judge(t, "persona", prof["trait"])
                        rec.update(persona=v, persona_raw=raw,
                                   persona_status=st,
                                   judge_version="v2-chat-nothink")
                        v, raw, st = judge(t, "fluency", "")
                        rec.update(fluency=v, fluency_raw=raw,
                                   fluency_status=st,
                                   judge_version="v2-chat-nothink")
                        if d in SIGN_DOSES:
                            v, raw, st = judge(t, "bipolar", prof["trait"])
                            rec.update(bipolar=v, bipolar_raw=raw,
                                       bipolar_status=st,
                                       judge_version="v2-chat-nothink")
                    append_jsonl(raw_p, rec)
                    done.add(key)
                    n_new += 1
            log(f"prompt {p['id']} done ({n_new} new this session)")
    except KeyboardInterrupt:
        log("interrupted; progress saved, rerun to resume")
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    man = json.load(open(man_p))
    man.update(state="running",
               counts={"new_this_session": n_new},
               peak_gpu_gb=peak, checkpoint_commit=rev,
               calibration=[{k: c[k] for k in
                             ("profile", "layer", "neuron", "dose",
                              "median", "source", "eligible", "push", "w_norm")}
                            for c in cal])
    atomic_json(man_p, man)
    log(f"new records this session: {n_new}; peak GPU GB: {peak}")
    log("Aggregate anytime (no GPU):  !python colab_full.py aggregate")


def do_aggregate(man_p, raw_p, cfg):
    recs, bad = read_jsonl(raw_p)
    ok = [r for r in recs if r.get("config_sha") == cfg["config_sha"]
          and r.get("live") is True and r.get("status") == "ok"
          and r.get("text", "").strip()]
    if not ok:
        log("no live records yet; run MODE=run first")
        sys.exit(5)
    for r in ok:
        if r.get("mock") is True or r.get("label") == "pre-recorded":
            raise ValueError("synthetic row refused: " + str(r.get("condition")))
    dose_rows = []
    for name in sorted(PROFILES):
        for d in DOSE_GRID:
            rows = [r for r in ok if r.get("profile") == name
                    and r.get("dose") == d]
            ps = [r["persona"] for r in rows
                  if isinstance(r.get("persona"), int)]
            fl = [r["fluency"] for r in rows
                  if isinstance(r.get("fluency"), int)]
            ks = [r.get("keyword_score") for r in rows
                  if isinstance(r.get("keyword_score"), (int, float))]
            ppls = [r["ppl"] for r in rows
                    if isinstance(r.get("ppl"), (int, float))]
            kls = [r["kl"] for r in rows
                   if isinstance(r.get("kl"), (int, float))]
            fks = [r["fork"] for r in rows
                   if isinstance(r.get("fork"), int)]
            ci = boot_mean_ci(ps)
            cpr = sum(1 for r in rows
                      if isinstance(r.get("persona"), int)
                      and isinstance(r.get("fluency"), int)
                      and (r["persona"] - 1) / 4 >= 0.75 and r["fluency"] >= 4)
            dose_rows.append({"profile": name, "dose": d,
                              "ps": ci, "n_requested": len(rows),
                              "keyword_mean": round(sum(ks) / len(ks), 4) if ks else None,
                              "fluency_mean": round(sum(fl) / len(fl), 3) if fl else None,
                              "cpr_count": cpr,
                              "ppl_median": sorted(ppls)[len(ppls) // 2] if ppls else None,
                              "kl_median": sorted(kls)[len(kls) // 2] if kls else None,
                              "fork_median": sorted(fks)[len(fks) // 2] if fks else None,
                              "fork_identical_frac": round(sum(1 for f in fks if f == 151) / len(fks), 3) if fks else None})
    sign_rows, all_p = [], []
    for name in sorted(PROFILES):
        g = {}
        for d in SIGN_DOSES:
            g[d] = [r["bipolar"] / 2 for r in ok
                    if r.get("profile") == name and r.get("dose") == d
                    and isinstance(r.get("bipolar"), int)]
        pn = signflip_p(g[-0.8], g[0.0])["p"] if g[-0.8] and g[0.0] else float("nan")
        pp = signflip_p(g[0.8], g[0.0])["p"] if g[0.8] and g[0.0] else float("nan")
        all_p += [pn, pp]
        sign_rows.append({"profile": name,
                          "means": {str(d): (round(sum(v) / len(v), 3) if v else None)
                                    for d, v in g.items()},
                          "n": {str(d): len(v) for d, v in g.items()},
                          "p_neg_vs_zero": pn, "p_pos_vs_zero": pp})
    out = {"run_dir": RUN_DIR, "config_sha": cfg["config_sha"],
           "n_records": len(recs), "n_usable": len(ok),
           "corrupt_lines": bad, "dose_table": dose_rows,
           "sign_table": sign_rows, "sign_holm": holm(all_p)}
    atomic_json(os.path.join(RUN_DIR, "aggregates.json"), out)
    log(f"aggregated {len(ok)}/{len(recs)} usable")
    log("==================== PASTE-BACK START ====================")
    print(json.dumps(out, indent=1))
    log("==================== PASTE-BACK END ======================")


if __name__ == "__main__":
    main()
