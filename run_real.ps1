# REAL mode — needs GPU + HF access
# pip install -r requirements.txt  (CUDA torch)
$env:MOCK_MODE="0"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn backend.app:app --port 8000"
Start-Sleep -Seconds 2
Set-Location frontend
npm run dev
