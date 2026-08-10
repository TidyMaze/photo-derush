#!/usr/bin/env python3
"""
Automated Deployment & Verification Script for Photo-Derush Darktable Plugin.
Usage:
    python deploy.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Force UTF-8 output encoding if supported
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Paths
ROOT_DIR = Path(__file__).parent.resolve()
LUA_FILE = ROOT_DIR / "lua" / "derush.lua"
CLI_BRIDGE = ROOT_DIR / "src" / "cli_bridge.py"
TRAINING_CORE = ROOT_DIR / "src" / "training_core.py"

CLIPPY_PNG = ROOT_DIR / "lua" / "clippy.png"

LOCAL_APPDATA = os.environ.get("LOCALAPPDATA", "")
TARGET_DIR = Path(LOCAL_APPDATA) / "darktable" / "lua" / "derush" if LOCAL_APPDATA else None

def main():
    print("========================================")
    print(" Photo-Derush Plugin Deployer")
    print("========================================")

    # Step 1: Lua Syntax Check
    print("\n1. Checking Lua Syntax (luaparser)...")
    try:
        from luaparser import ast
        code = LUA_FILE.read_text(encoding="utf-8")
        ast.parse(code)
        print("   [OK] Lua Syntax Check: 100% PASSED")
    except Exception as e:
        print(f"   [FAIL] Lua Syntax Check FAILED: {e}")
        sys.exit(1)

    # Step 2: Pytest Integration Suite
    print("\n2. Running Integration Tests (pytest)...")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_cli_bridge_integration.py", "-q"], cwd=ROOT_DIR)
    if result.returncode != 0:
        print("   [FAIL] Integration Tests FAILED! Aborting deployment.")
        sys.exit(1)
    print("   [OK] Integration Tests: PASSED")

    # Step 3: Copy Files to Darktable Plugin Directory
    if not TARGET_DIR:
        print("   [FAIL] LOCALAPPDATA environment variable not found.")
        sys.exit(1)

    print(f"\n3. Deploying Plugin to {TARGET_DIR}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    files_to_deploy = [
        (LUA_FILE, TARGET_DIR / "derush.lua"),
        (CLI_BRIDGE, TARGET_DIR / "cli_bridge.py"),
        (TRAINING_CORE, TARGET_DIR / "training_core.py"),
        (CLIPPY_PNG, TARGET_DIR / "clippy.png"),
    ]

    for src, dst in files_to_deploy:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"   -> Deployed: {src.relative_to(ROOT_DIR)} -> {dst}")

    print("\n========================================")
    print(" SUCCESS! Plugin deployed to Darktable!")
    print(f" Target Folder: {TARGET_DIR}")
    print("========================================\n")

if __name__ == "__main__":
    main()
