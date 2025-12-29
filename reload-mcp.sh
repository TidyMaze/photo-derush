#!/bin/bash
# Reload MCP servers by updating the config file with a new timestamp

MCP_CONFIG="$HOME/.cursor/mcp.json"

if [ ! -f "$MCP_CONFIG" ]; then
    echo "Error: MCP config file not found at $MCP_CONFIG"
    exit 1
fi

echo "Updating MCP config file to trigger reload..."

# Use Python to properly update the JSON file
python3 << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

mcp_config = Path.home() / ".cursor" / "mcp.json"

try:
    # Read existing config
    with open(mcp_config, 'r') as f:
        config = json.load(f)
    
    # Update or add _RELOAD_TIMESTAMP to screenshot section
    if "mcpServers" in config and "screenshot" in config["mcpServers"]:
        import time
        timestamp = str(int(time.time()))
        
        if "env" not in config["mcpServers"]["screenshot"]:
            config["mcpServers"]["screenshot"]["env"] = {}
        
        config["mcpServers"]["screenshot"]["env"]["_RELOAD_TIMESTAMP"] = timestamp
        
        # Write back with proper formatting
        with open(mcp_config, 'w') as f:
            json.dump(config, f, indent=2)
            f.write('\n')  # Ensure newline at end
        
        print(f"✅ MCP config file updated: {mcp_config}")
        print(f"   Reload timestamp: {timestamp}")
        print("")
        print("Note: Cursor should detect the change and reload MCP servers automatically.")
        print("If not, you may need to restart Cursor.")
        sys.exit(0)
    else:
        print("❌ 'screenshot' section not found in MCP config")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error updating MCP config: {e}")
    sys.exit(1)
PYTHON_SCRIPT

exit_code=$?
if [ $exit_code -ne 0 ]; then
    exit $exit_code
fi
