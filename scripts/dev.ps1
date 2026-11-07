# Development mode: FastAPI with auto-reload on :8000 and the Vite dev server on :5173 (proxying /api).
param([ValidateSet("auto", "cuda", "cpu")][string]$Torch = "auto")
$root = Split-Path -Parent $PSScriptRoot
if ($Torch -eq "auto") { $Torch = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" } }
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; uv run --extra $Torch uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
Write-Host "Backend: http://127.0.0.1:8000/docs   Frontend: http://127.0.0.1:5173"
