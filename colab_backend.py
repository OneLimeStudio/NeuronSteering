# ===== COLAB CELL: run backend on Colab GPU + expose via cloudflared =====
# 1. Runtime > T4/A100 GPU. 2. Paste this whole file into ONE cell. 3. Copy the public URL into frontend API box.
# !pip install -q fastapi uvicorn transformers accelerate cloudflared 2>&1 | tail -1
# !curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared && chmod +x /tmp/cloudflared

import subprocess, time, threading, re, os
os.environ["MOCK_MODE"] = "0"  # live GPU

def _run():
    subprocess.run(["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"], cwd="/content/project")

th = threading.Thread(target=_run, daemon=True)
th.start()
time.sleep(12)
# open tunnel
proc = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://localhost:8000"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print("Waiting for cloudflared URL… (30s)")
for line in proc.stdout:  # type: ignore
    print(line.strip())
    m = re.search(r"https://[\w\-]+\.trycloudflare\.com", line)
    if m:
        print("\n✅ BACKEND URL:", m.group(0))
        print("Paste it + /api/status into the frontend API box, e.g.", m.group(0))
        break
