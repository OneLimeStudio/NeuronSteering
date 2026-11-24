"""Runs every experiment and saves to demo_cache/*.json. Uses live model if available, else mock."""
import os, json, sys
os.environ.setdefault("MOCK_MODE", os.getenv("MOCK_MODE", "1"))
from fastapi.testclient import TestClient
from backend.app import app

c = TestClient(app)
PROMPT = "What should I pack for a beach trip?"
out = {}
out["status"] = c.get("/api/status").json()
out["profiles"] = c.get("/api/profiles").json()
out["compare_jesus"] = c.post("/api/compare", json={"prompt": PROMPT, "profile": "jesus", "max_tokens": 150, "seed": 0}).json()
out["compare_baseline"] = c.post("/api/compare", json={"prompt": PROMPT, "profile": "baseline", "max_tokens": 150, "seed": 0}).json()
out["dose_jesus"] = c.post("/api/dose-sweep", json={"prompt": PROMPT, "layer": 15, "neuron": 4041, "seed": 0, "max_tokens": 120, "doses": [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.8,1.0], "keywords": ["jesus","god","lord","bible","faith"]}).json()
out["xray"] = c.post("/api/xray", json={"prompt": PROMPT, "layer": 15, "neuron": 4041, "dose": 0.8}).json()
out["logits"] = c.post("/api/logits", json={"prompt": PROMPT, "layer": 15, "neuron": 4041, "dose": 0.8}).json()
out["random_ctrl"] = c.post("/api/controls/random", json={"prompt": PROMPT, "layer": 15, "neuron": 4041, "dose": 0.8}).json()
out["sign_ctrl"] = c.post("/api/controls/sign", json={"prompt": PROMPT, "layer": 15, "neuron": 4041, "dose": 0.8}).json()
out["scale_ctrl"] = c.post("/api/controls/scale", json={"prompt": PROMPT}).json()
out["ablation"] = c.post("/api/controls/ablation", json={"prompt": PROMPT}).json()
out["gallery"] = c.post("/api/gallery", json={"prompt": PROMPT, "max_tokens": 150, "seed": 0}).json()

os.makedirs("demo_cache", exist_ok=True)
for k, v in out.items():
    with open(f"demo_cache/{k}.json", "w") as f:
        json.dump(v, f, indent=2)
    print(f"saved demo_cache/{k}.json")
print("DONE. Mock=" + str(out["status"].get("mock")))
