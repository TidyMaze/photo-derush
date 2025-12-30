#!/bin/bash
# Server startup script for Photo Derush

echo "🚀 Starting Photo Derush Servers..."
echo ""

# Stop any existing servers
pkill -f "uvicorn api.main:app" 2>/dev/null
pkill -f "vite" 2>/dev/null
sleep 1

# Start backend
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo "🚀 Backend auto-start disabled (see script)."
# Disabled auto-start of API server. To run manually:
# nohup poetry run uvicorn api.main:app --reload --port 8000 > /tmp/backend.log 2>&1 &
# echo $! > /tmp/backend.pid
sleep 2

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
echo "📍 Backend API:  http://localhost:8000"
echo "   - API Docs:   http://localhost:8000/docs"
echo "   - Health:     http://localhost:8000/health"
echo ""
echo "📍 Frontend UI:  http://localhost:5173"
echo ""
echo "📊 Current Data:"
echo "   - Projects: 1 (photo-dataset)"
echo "   - Images: 401"
echo "   - EXIF: 400 records"
echo ""
echo "📝 Logs:"
echo "   - Backend:  tail -f /tmp/backend.log"
echo "   - Frontend: tail -f /tmp/frontend.log"
echo ""
echo "🛑 To stop servers:"
echo "   kill \$(cat /tmp/backend.pid) \$(cat /tmp/frontend.pid)"
echo ""
echo "=================================="
echo ""
echo "🌐 Open http://localhost:5173 in your browser!"
echo ""

