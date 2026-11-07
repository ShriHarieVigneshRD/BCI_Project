# EEGNet Architecture Playground - one-command local start (Windows PowerShell).
# Installs pinned dependencies, builds the frontend once and serves everything at http://127.0.0.1:8000
param([ValidateSet("auto", "cuda", "cpu")][string]$Torch = "auto", [switch]$Rebuild)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if ($Torch -eq "auto") {
    $Torch = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" }
}
Write-Host "PyTorch build: $Torch"

Push-Location "$root\backend"
uv sync --extra $Torch
Pop-Location

if ($Rebuild -or -not (Test-Path "$root\frontend\dist\index.html")) {
    Push-Location "$root\frontend"
    npm ci
    npm run build
    Pop-Location
}

Push-Location "$root\backend"
Write-Host "Open http://127.0.0.1:8000 (Ctrl+C to stop)"
uv run --extra $Torch python -m app.main
Pop-Location
