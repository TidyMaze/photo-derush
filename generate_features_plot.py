#!/usr/bin/env python3
"""
Example: Generate top features correlation plot

This script demonstrates how to use plot_top_features_correlation.py
"""

import os
import subprocess
import sys

# Best-effort early mitigation for native libs
try:
    import src.init_mitigation  # noqa: F401
except Exception:
    pass


def main():
    script_path = os.path.join(os.path.dirname(__file__), 'plot_top_features_correlation.py')

    print("=" * 70)
    print("TOP FEATURES CORRELATION PLOT GENERATOR")
    print("=" * 70)
    print()

    # Try auto-detection first
    print("Attempting to auto-detect image directory from app config...")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False
    )

    if result.returncode == 0:
        print()
        print("✓ Plot generated successfully!")
        print("  Output: plots/top_features_correlation.png")
    else:
        print()
        print("✗ Could not generate plot automatically.")
        print()
        print("Usage examples:")
        print(f"  python {script_path} ~/Pictures")
        print(f"  python {script_path} ~/Downloads --output correlation.png")
        print(f"  python {script_path} --no-model")
        print()
        print(f"Or run: python {script_path} --help")

if __name__ == '__main__':
    main()

