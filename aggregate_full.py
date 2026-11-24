"""Local aggregation for a finished Colab full-study run (CPU only).

You download TWO files from Drive and place them here; this script does the rest:
  Drive: /content/drive/MyDrive/steering_runs/full-dose50/raw_generations.jsonl
         /content/drive/MyDrive/steering_runs/full-dose50/manifest.json
  Local: runs/full-dose50/raw_generations.jsonl
         runs/full-dose50/manifest.json   (optional, used for provenance)

Run:  python aggregate_full.py [run_dir, default runs/full-dose50]

Writes aggregates.json next to the inputs (atomic) and prints a summary.
Statistics come from backend/stats.py (full CI endpoints, never half-widths).
Synthetic rows (mock True / pre-recorded) are refused. Partial runs aggregate
fine; missing rows are counted, never silently dropped.
"""

import json
import os
import statistics
import sys

from backend.runner import atomic_write_json
from backend.stats import bootstrap_mean_ci, holm_adjust, paired_signflip_pvalue

DOSE_GRID = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
             0.25, 0.5, 0.8, 1.0, 1.25, 1.5]
SIGN_DOSES = [-0.8, 0.0, 0.8]
PROFILES = ("jesus", "yandere", "suspicious", "reality")


def median(vals):
    return statistics.median(vals) if vals else None


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("runs", "full-dose50")
    raw_p = os.path.join(run_dir, "raw_generations.jsonl")
    if not os.path.exists(raw_p):
        print(f"missing {raw_p}: download raw_generations.jsonl from Drive into {run_dir}/")
        sys.exit(2)
    recs, corrupt = [], 0
    with open(raw_p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                corrupt += 1
    shas = {r.get("config_sha") for r in recs}
    ok = [r for r in recs if r.get("live") is True and r.get("status") == "ok"
          and (r.get("text") or "").strip()]
    for r in ok:
        if r.get("mock") is True or r.get("label") == "pre-recorded":
            raise ValueError("synthetic row refused: " + str(r.get("condition")))
    print(f"records: {len(recs)} total, {len(ok)} usable live, "
          f"{corrupt} corrupt lines, config_shas: {len(shas)}")

    dose_rows = []
    for name in PROFILES:
        for d in DOSE_GRID:
            rows = [r for r in ok if r.get("profile") == name and r.get("dose") == d]
            ps = [r["persona"] for r in rows if isinstance(r.get("persona"), int)]
            fl = [r["fluency"] for r in rows if isinstance(r.get("fluency"), int)]
            ks = [r["keyword_score"] for r in rows
                  if isinstance(r.get("keyword_score"), (int, float))]
            ppls = [r["ppl"] for r in rows if isinstance(r.get("ppl"), (int, float))]
            kls = [r["kl"] for r in rows if isinstance(r.get("kl"), (int, float))]
            fks = [r["fork"] for r in rows if isinstance(r.get("fork"), int)]
            cpr = sum(1 for r in rows
                      if isinstance(r.get("persona"), int)
                      and isinstance(r.get("fluency"), int)
                      and (r["persona"] - 1) / 4 >= 0.75 and r["fluency"] >= 4)
            dose_rows.append({"profile": name, "dose": d,
                              "ps": bootstrap_mean_ci(ps, unit="prompt"),
                              "n_requested": len(rows),
                              "keyword_mean": round(sum(ks) / len(ks), 4) if ks else None,
                              "fluency_mean": round(sum(fl) / len(fl), 3) if fl else None,
                              "cpr_count": cpr,
                              "ppl_median": median(sorted(ppls)),
                              "kl_median": median(sorted(kls)),
                              "fork_median": median(sorted(fks)),
                              "fork_identical_frac": round(sum(1 for f in fks if f == 151) / len(fks), 3) if fks else None})

    sign_rows, fam = [], []
    for name in PROFILES:
        g = {}
        for d in SIGN_DOSES:
            g[d] = [r["bipolar"] / 2 for r in ok
                    if r.get("profile") == name and r.get("dose") == d
                    and isinstance(r.get("bipolar"), int)]
        pn = paired_signflip_pvalue(g[-0.8], g[0.0])
        pp = paired_signflip_pvalue(g[0.8], g[0.0])
        fam += [pn["p"], pp["p"]]
        sign_rows.append({"profile": name,
                          "bipolar_means": {str(d): (round(sum(v) / len(v), 3) if v else None)
                                            for d, v in g.items()},
                          "n": {str(d): len(v) for d, v in g.items()},
                          "neg_vs_zero": pn, "pos_vs_zero": pp})
    adj = holm_adjust([p for p in fam if p == p])
    it = iter(adj)
    for r in sign_rows:
        for k in ("neg_vs_zero", "pos_vs_zero"):
            if r[k]["p"] == r[k]["p"]:
                r[k]["p_holm"] = next(it)

    man_p = os.path.join(run_dir, "manifest.json")
    provenance = {}
    if os.path.exists(man_p):
        m = json.load(open(man_p, encoding="utf-8"))
        provenance = {"run_id": m.get("run_id"), "config_sha": m.get("config_sha"),
                      "checkpoint_commit": m.get("checkpoint_commit"),
                      "peak_gpu_gb": m.get("peak_gpu_gb"),
                      "calibration": m.get("calibration")}
    out = {"run_dir": run_dir, "provenance": provenance,
           "n_records": len(recs), "n_usable": len(ok),
           "corrupt_lines": corrupt, "config_shas": sorted(s for s in shas if s),
           "dose_table": dose_rows, "sign_table": sign_rows}
    ap = os.path.join(run_dir, "aggregates.json")
    atomic_write_json(ap, out)
    complete = sum(1 for r in dose_rows if r["ps"]["n_valid"] > 0)
    print(f"dose cells with data: {complete}/{len(dose_rows)}; wrote {ap}")
    for r in dose_rows:
        if r["ps"]["n_valid"]:
            print(f"  {r['profile']:10s} d={r['dose']:+5.2f} "
                  f"PS={r['ps']['mean']:.2f} [{r['ps']['lo']:.2f},{r['ps']['hi']:.2f}] "
                  f"n={r['ps']['n_valid']}")


if __name__ == "__main__":
    main()
