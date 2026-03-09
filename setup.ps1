# EXXA Setup Script for Windows PowerShell
# Run with: .\setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  EXXA Project Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  [INFO] Virtual environment already exists. Skipping..." -ForegroundColor Yellow
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Virtual environment created: ./venv" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "  [OK] Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "  [OK] pip upgraded" -ForegroundColor Green

# Install core requirements
Write-Host ""
Write-Host "[5/6] Installing core dependencies..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Gray
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Core dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Test installation
Write-Host ""
Write-Host "[6/6] Testing installation..." -ForegroundColor Yellow
python -c "import torch; import numpy as np; print('PyTorch:', torch.__version__); print('NumPy:', np.__version__); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Installation test passed" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Installation test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Activate environment: .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  2. Choose a subproject to work on" -ForegroundColor Gray
Write-Host "  3. Install subproject-specific requirements" -ForegroundColor Gray
Write-Host "  4. See SETUP.md for detailed instructions" -ForegroundColor Gray
Write-Host ""
