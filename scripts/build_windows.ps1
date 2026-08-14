$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$modelCache = Join-Path $env:LOCALAPPDATA 'ModularRAG\models\huggingface'
$env:HF_HOME = $modelCache
$env:HUGGINGFACE_HUB_CACHE = $modelCache
& .\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --name ModularRAGLauncher `
  --collect-all streamlit --collect-all qdrant_client --collect-all sentence_transformers `
  --add-data "config;config" --add-data "src\observability\dashboard;src\observability\dashboard" `
  src\production\launcher.py
