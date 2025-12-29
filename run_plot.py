#!/usr/bin/env python3
"""Simple wrapper to run the correlation plot with better error handling."""
import os
import sys

# Early mitigation (best-effort) to avoid native thread storms when plotting
try:
    import src.init_mitigation  # noqa: F401
except Exception:
    pass

# Make sure we're in the right directory
os.chdir('/Users/yannrolland/work/photo-derush')
sys.path.insert(0, '.')

def run_plot():
    """Run the plot generation."""
    import logging
    logging.basicConfig(level=logging.INFO)

    from plot_top_features_correlation import main
    try:
        main()
        return 0
    except SystemExit as e:
        return e.code if e.code is not None else 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(run_plot())

