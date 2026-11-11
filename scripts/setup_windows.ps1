# One-time setup on Windows. Run from the project folder in PowerShell:
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#     .\scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
py -3.10 -m venv venv
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --default-timeout=300 --retries 10 -r requirements.txt

$dl = Read-Host "Install the deep-learning extras (TensorFlow, OpenCV; large download; needed for camera, fire CNN, faces)? [y/N]"
if ($dl -eq "y") {
    pip install --default-timeout=300 --retries 10 -r requirements-ml.txt
    python scripts/download_face_models.py
}

Write-Host "Training the ML models (about a minute)..."
python -m ml.train_ml

Write-Host "Create the first dashboard administrator (password: 8+ characters, not the username):"
python -m src.users add admin --role admin

Write-Host ""
Write-Host "Setup finished. Start the dashboard with:  .\scripts\run_dashboard.bat --sim --scenario gas_leak"
