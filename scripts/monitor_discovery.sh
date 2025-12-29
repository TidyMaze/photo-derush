#!/bin/bash
# Monitor feature discovery script progress

PID=$(ps aux | grep "discover_new_features.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ Feature discovery script is not running"
    exit 1
fi

echo "=========================================="
echo "Feature Discovery Progress Monitor"
echo "=========================================="
echo ""

# Process info
echo "📊 Process Status:"
ps -p $PID -o pid,etime,%cpu,%mem,command 2>/dev/null || echo "Process finished"
echo ""

# Check results file
if [ -f .cache/feature_discovery_results.json ]; then
    echo "✅ Results file exists!"
    echo ""
    echo "📄 Results Summary:"
    python3 -c "
import json
import sys
try:
    with open('.cache/feature_discovery_results.json') as f:
        data = json.load(f)
    
    if 'baseline' in data:
        baseline = data['baseline']['metrics']
        print(f\"  Baseline Accuracy: {baseline['accuracy']:.4f}\")
        print(f\"  Baseline F1: {baseline['f1']:.4f}\")
    
    if 'ranked_improvements' in data and data['ranked_improvements']:
        print(f\"\\n  Feature Groups Tested: {len(data['ranked_improvements'])}\")
        if len(data['ranked_improvements']) > 0:
            best = data['ranked_improvements'][0]
            print(f\"  Best Improvement: {best['group']} (+{best['accuracy_diff']*100:.2f}%)\")
    
    if 'helpful_groups' in data and data['helpful_groups']:
        print(f\"\\n  Helpful Groups: {', '.join(data['helpful_groups'])}\")
except Exception as e:
    print(f\"  Error reading results: {e}\")
" 2>/dev/null || echo "  (Results file exists but may be incomplete)"
else
    echo "⏳ Results file not created yet - script still running..."
    echo ""
    echo "💡 Estimated progress based on elapsed time:"
    ELAPSED=$(ps -p $PID -o etime= 2>/dev/null | awk '{print $1}' | sed 's/://' | awk '{if(NF==2) print $1*60+$2; else print $1}')
    if [ ! -z "$ELAPSED" ]; then
        echo "  Elapsed: ${ELAPSED} seconds"
        if [ "$ELAPSED" -lt 60 ]; then
            echo "  Likely in: Phase 1 (Building dataset) or Phase 2 (Baseline training)"
        elif [ "$ELAPSED" -lt 300 ]; then
            echo "  Likely in: Phase 3 (Testing feature groups)"
        else
            echo "  Likely in: Phase 4 (Combining features) or Phase 5 (Saving results)"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Monitor again: ./scripts/monitor_discovery.sh"
echo "=========================================="

