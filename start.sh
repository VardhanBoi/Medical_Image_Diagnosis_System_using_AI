#!/bin/bash
# ── MedVision AI — Start Script ────────────────────────────────────────────────
# Run from the project root:  bash start.sh

set -e

echo ""
echo "  🔬  MedVision AI — Medical Image Diagnosis System"
echo "  ──────────────────────────────────────────────────"
echo ""

# 1. Install / verify dependencies
echo "  [1/3] Installing dependencies..."
pip install -q -r requirements.txt

# 2. Start Flask backend
echo ""
echo "  [2/3] Starting Flask backend on http://localhost:5000 ..."
echo ""
python app.py &
BACKEND_PID=$!
echo "        Backend PID: $BACKEND_PID"

# 3. Open frontend
echo ""
echo "  [3/3] Frontend ready at: frontend/index.html"
echo "        Open it in your browser, or run:"
echo "        python -m http.server 8080 --directory frontend"
echo ""
echo "  ──────────────────────────────────────────────────"
echo "  Press Ctrl+C to stop the backend."
echo ""

# Keep alive until user stops it
trap "kill $BACKEND_PID 2>/dev/null; echo '  Backend stopped.'" EXIT
wait $BACKEND_PID