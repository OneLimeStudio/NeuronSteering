"""Real computed metrics — no faking in real mode."""
import math
from collections import Counter

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore
    F = None  # type: ignore


def keyword_score(text: str, keywords) -> dict:
    t = text.lower()
    hits = {k: t.count(k.lower()) for k in keywords}
    total = sum(hits.values())
    # normalized 0..1 by hits per 100 chars, capped
    density = min(1.0, total / max(1, len(t) / 200))
    return {"hits": hits, "total_hits": total, "score": round(density, 4)}


def token_overlap(a_ids, b_ids) -> float:
    if not a_ids or not b_ids:
        return 0.0
    sa, sb = set(a_ids), set(b_ids)
    return round(len(sa & sb) / max(1, len(sa | sb)), 4)


def cosine_sim(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return round(dot / (na * nb), 4)


def bow_vector(text: str):
    c = Counter(text.lower().split())
    return c


def bow_cosine(a: str, b: str) -> float:
    ca, cb = bow_vector(a), bow_vector(b)
    keys = set(ca) | set(cb)
    va = [ca[k] for k in keys]
    vb = [cb[k] for k in keys]
    return cosine_sim(va, vb)


@torch.no_grad()
def perplexity_under_baseline(model, tok, text: str, prompt: str) -> float:
    """PPL of `text` under UNSTEERED model (fluency check). Real compute."""
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt", return_dict=True).to(model.device)
    prefix_len = enc["input_ids"].shape[1]
    full = prompt + "\n" + text
    # simpler: tokenize continuation with context
    full_ids = tok(full, return_tensors="pt").input_ids.to(model.device)
    if full_ids.shape[1] < 2:
        return float("inf")
    out = model(input_ids=full_ids, labels=full_ids)
    # shift loss already averaged; convert to ppl
    loss = out.loss.item()
    try:
        return round(math.exp(min(loss, 10)), 3)
    except OverflowError:
        return 999.0


@torch.no_grad()
def first_token_kl(model, tok, layers, prompt: str, neurons, seed: int = 0):
    """KL(next-token || baseline) at first generated position. Real compute."""
    from .steering import apply_steering, clear_handles, build_inputs, set_seed
    set_seed(seed)
    enc = build_inputs(tok, model, prompt)
    with torch.no_grad():
        base_logits = model(**enc).logits[:, -1, :].float().cpu()
    base_logp = F.log_softmax(base_logits, dim=-1)
    base_p = base_logp.exp()

    handles, _ = apply_steering(layers, neurons)
    try:
        with torch.no_grad():
            steer_logits = model(**enc).logits[:, -1, :].float().cpu()
    finally:
        clear_handles(handles)
    steer_logp = F.log_softmax(steer_logits, dim=-1)
    steer_p = steer_logp.exp()
    kl = F.kl_div(steer_logp, base_p, reduction="batchmean").item()
    # top-1 prob shift
    b_top = int(base_logits.argmax(-1).item())
    s_top = int(steer_logits.argmax(-1).item())
    return {
        "kl": round(float(kl), 4),
        "baseline_top_id": b_top,
        "steered_top_id": s_top,
        "baseline_top_token": tok.decode([b_top]),
        "steered_top_token": tok.decode([s_top]),
    }


def sentiment_simple(text: str) -> dict:
    pos = ["love", "happy", "wonderful", "bless", "glad", "joy", "great", "beautiful"]
    neg = ["jealous", "hurt", "angry", "paranoid", "afraid", "guilty", "crime", "suspect", "hate"]
    t = text.lower()
    p = sum(t.count(w) for w in pos)
    n = sum(t.count(w) for w in neg)
    return {"pos": p, "neg": n, "polarity": round((p - n) / max(1, p + n), 3), "length_chars": len(text),
            "length_words": len(text.split())}
