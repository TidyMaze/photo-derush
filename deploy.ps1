# PowerShell Wrapper for deploy.py
$VENV_PYTHON = "$env:USERPROFILE\AppData\Local\pypoetry\Cache\virtualenvs\photo-app-rBz6-pE0-py3.12\Scripts\python.exe"
& $VENV_PYTHON "$PSScriptRoot\deploy.py"
