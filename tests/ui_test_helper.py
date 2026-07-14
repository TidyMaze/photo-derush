#!/usr/bin/env python3
"""
ui_test_helper.py - Tools for screenshot-based UI testing of photo-derush.
Usage:
    python ui_test_helper.py screenshot         # capture and save screenshot
    python ui_test_helper.py click X Y          # click at screen coordinates
    python ui_test_helper.py move X Y           # move mouse to coordinates
    python ui_test_helper.py type "text"        # type text
    python ui_test_helper.py key KEYNAME        # press a key (e.g. return, space)
    python ui_test_helper.py window             # print app window bounds
"""
import subprocess
import sys
import os
import json
from pathlib import Path

SCREENSHOT_PATH = "/tmp/photo_derush_screenshot.png"
APP_NAME = "Python"  # The app runs as Python on macOS


def screenshot(output_path=SCREENSHOT_PATH):
    """Take full-screen screenshot, return path."""
    subprocess.run(["screencapture", "-x", output_path], check=True)
    print(f"Screenshot saved: {output_path}")
    return output_path


def get_window_bounds():
    """Return {x, y, w, h} of the app window using osascript."""
    script = f"""
tell application "System Events"
    set proc to first process whose name is "{APP_NAME}"
    set win to first window of proc
    set {{x, y}} to position of win
    set {{w, h}} to size of win
    return x & "," & y & "," & w & "," & h
end tell
"""
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return None
    parts = result.stdout.strip().split(", ")
    return {"x": int(parts[0]), "y": int(parts[1]), "w": int(parts[2]), "h": int(parts[3])}


def click(x, y, button="left"):
    """Click at screen coordinates."""
    flag = "c:" if button == "left" else "rc:"
    subprocess.run(["cliclick", f"{flag}{x},{y}"], check=True)
    print(f"Clicked ({x}, {y})")


def move(x, y):
    """Move mouse to coordinates."""
    subprocess.run(["cliclick", f"m:{x},{y}"], check=True)


def type_text(text):
    """Type text."""
    subprocess.run(["cliclick", f"t:{text}"], check=True)


def key_press(key):
    """Press a key by name (e.g. return, space, escape, tab)."""
    subprocess.run(["cliclick", f"kp:{key}"], check=True)


def focus_app():
    """Bring the app window to front."""
    script = f'tell application "{APP_NAME}" to activate'
    subprocess.run(["osascript", "-e", script])


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    if cmd == "screenshot":
        path = args[1] if len(args) > 1 else SCREENSHOT_PATH
        screenshot(path)
    elif cmd == "click":
        click(int(args[1]), int(args[2]))
    elif cmd == "move":
        move(int(args[1]), int(args[2]))
    elif cmd == "type":
        type_text(args[1])
    elif cmd == "key":
        key_press(args[1])
    elif cmd == "window":
        bounds = get_window_bounds()
        print(json.dumps(bounds, indent=2))
    elif cmd == "focus":
        focus_app()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
