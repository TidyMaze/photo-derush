#!/bin/bash
# Server startup script for Photo Derush

echo "🚀 Starting Photo Derush Servers..."
echo ""

# Stop any existing servers
pkill -f "vite" 2>/dev/null
sleep 1

# Change to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Start frontend
cd "$SCRIPT_DIR/web"
echo "🎨 Starting Frontend UI..."
nohup npm run dev > /tmp/frontend.log 2>&1 &
echo $! > /tmp/frontend.pid
sleep 3

# Verify
echo ""
echo "=================================="
echo "✅ SERVERS STARTED"
echo "=================================="
echo ""
echo "📍 Frontend UI:  http://localhost:5173"
echo ""
echo "📝 Logs:"
echo "   - Frontend: tail -f /tmp/frontend.log"
echo ""
echo "🛑 To stop servers:"
echo "   kill \$(cat /tmp/frontend.pid)"
echo ""
echo "=================================="
echo ""
echo "🌐 Open http://localhost:5173 in your browser!"
echo ""

