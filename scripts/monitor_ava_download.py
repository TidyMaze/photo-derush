#!/usr/bin/env python3
"""Monitor AVA image download progress."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

def monitor_download(output_dir: str, target: int = None, interval: int = 2):
    """Monitor download progress."""
    img_dir = Path(output_dir)
    
    print("AVA Download Monitor")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    if target:
        print(f"Target: {target} images")
    print("Press Ctrl+C to stop")
    print()
    
    prev_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Count images
            current = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
            
            # Calculate rate
            elapsed = time.time() - start_time
            if elapsed > 0:
                rate = (current - prev_count) / interval if prev_count > 0 else 0
                total_rate = current / elapsed if elapsed > 0 else 0
            else:
                rate = 0
                total_rate = 0
            
            # Estimate time remaining
            if target and total_rate > 0:
                remaining = (target - current) / total_rate
                eta_str = f"{int(remaining//60)}m {int(remaining%60)}s"
            else:
                eta_str = "N/A"
            
            # Display
            print(f"\r[{time.strftime('%H:%M:%S')}] {current:4d} images", end="")
            if target:
                print(f" ({current/target*100:5.1f}%)", end="")
            print(f" | Rate: {rate:5.1f} img/s | ETA: {eta_str}", end="", flush=True)
            
            prev_count = current
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"Final count: {current} images")


def main():
    parser = argparse.ArgumentParser(description="Monitor AVA download progress")
    parser.add_argument("--output-dir", default=".cache/ava_dataset/images", help="Image directory")
    parser.add_argument("--target", type=int, default=None, help="Target number of images")
    parser.add_argument("--interval", type=int, default=2, help="Update interval (seconds)")
    args = parser.parse_args()
    
    monitor_download(args.output_dir, args.target, args.interval)


if __name__ == "__main__":
    main()


