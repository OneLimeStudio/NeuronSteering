"""Annotation workflow: blinded export, label validation, no invented agreement."""

from backend.annotate import export_items, validate_labels


def _rec(rid, dose, text="hello world"):
    return {"record_id": rid, "status": "ok", "dose": dose, "prompt": "Say hi.",
            "text": text, "profile": "jesus", "run_id": "run-x"}


def test_export_withholds_conditions_and_is_reproducible():
    recs = [_rec(f"r{i}", d) for i, d in enumerate([0.0, 0.8] * 3)]
    a = export_items(recs, n=4, seed=0)
    b = export_items(recs, n=4, seed=0)
    assert a == b and len(a["items"]) == 4
    for item in a["items"]:
        assert set(item) == {"item_id", "prompt", "response"}
        assert item["item_id"].startswith("ann-")


def test_validate_labels_counts_problems_and_no_agreement():
    recs = [_rec("r1", 0.0), _rec("r2", 0.8)]
    exp = export_items(recs, n=2, seed=0)
    ids = [i["item_id"] for i in exp["items"]]
    out = validate_labels([{"item_id": ids[0], "label": 4},
                           {"item_id": ids[0], "label": 4},
                           {"item_id": "ann-unknown", "label": 3},
                           {"item_id": ids[1], "label": 99}], ids, 1, 5)
    assert out["agreement"] is None
    assert out["n_valid"] == 1
    assert len(out["problems"]) == 3
