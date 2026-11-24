$env:MOCK_MODE="1"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn backend.app:app --port 8000"
Start-Sleep -Seconds 2
Set-Location frontend
npm run dev
