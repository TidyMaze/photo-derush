#!/bin/bash

# Debug Server Startup Script
# This script starts both servers and keeps them running

echo "════════════════════════════════════════════"
echo "  Photo Derush - Server Startup"
echo "════════════════════════════════════════════"
echo ""

# Change to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Kill any existing servers
echo "🛑 Stopping existing servers..."
pkill -9 -f "vite" 2>/dev/null
sleep 2

# Start frontend
echo "🚀 Starting Frontend UI on port 5173..."
cd web
npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

cd ..

# Wait and verify
sleep 4
echo ""
echo "════════════════════════════════════════════"
echo "  Verification"
echo "════════════════════════════════════════════"
echo ""

# Test frontend
if curl -s http://localhost:5173 | grep -q "root"; then
    echo "✅ Frontend: http://localhost:5173"
else
    echo "❌ Frontend: Not responding"
fi

echo ""
echo "════════════════════════════════════════════"
echo "  Ready!"
echo "════════════════════════════════════════════"
echo ""
echo "🌐 Open in browser: http://localhost:5173"
echo ""
echo "📝 To stop servers:"
echo "   kill $FRONTEND_PID"
echo ""
echo "📊 To view logs:"
echo "   Frontend: check terminal output"
echo ""
echo "⌨️  Press Ctrl+C to stop all servers"
echo ""

# Keep script running (trap Ctrl+C to kill servers)
trap "echo ''; echo 'Stopping servers...'; kill $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

# Wait for both processes
wait

