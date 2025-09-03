#!/bin/bash
set -euo pipefail

# end-lab.sh: Cleanly stop pod to halt GPU billing while leaving volume intact
# Echoes stopped state to avoid idle charges

echo "🛑 Stopping Cogit Research Lab Environment"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() { echo -e "${RED}ERROR: $1${NC}" >&2; exit 1; }
warn() { echo -e "${YELLOW}WARNING: $1${NC}" >&2; }
success() { echo -e "${GREEN}✓ $1${NC}"; }
info() { echo "ℹ️  $1"; }

# Load environment variables
if [[ -f ".env" ]]; then
    source .env
else
    error ".env file not found. Cannot determine pod ID. Run genesis.sh and start-lab.sh first."
fi

# Check if POD_ID is available
if [[ -z "${POD_ID:-}" ]]; then
    warn "POD_ID not found in .env. Attempting to find running pods..."
    
    # Look for running pods
    RUNNING_PODS=$(runpodctl get pods --output json 2>/dev/null | jq -r '.[] | select(.status == "RUNNING") | .id' | head -n1)
    
    if [[ -n "$RUNNING_PODS" ]]; then
        POD_ID="$RUNNING_PODS"
        info "Found running pod: $POD_ID"
    else
        error "No running pods found and POD_ID not in .env"
    fi
fi

# Verify pod exists and get current status
echo -e "\n🔍 Checking pod status..."
POD_STATUS=$(runpodctl get pod "$POD_ID" --output json 2>/dev/null | jq -r '.status' || echo "NOT_FOUND")

if [[ "$POD_STATUS" == "NOT_FOUND" ]]; then
    error "Pod $POD_ID not found. It may have already been terminated."
elif [[ "$POD_STATUS" == "TERMINATED" ]]; then
    warn "Pod $POD_ID is already terminated."
    echo "Billing should already be stopped."
    exit 0
elif [[ "$POD_STATUS" != "RUNNING" ]]; then
    warn "Pod $POD_ID is in status: $POD_STATUS"
fi

# Get pod information before stopping
POD_INFO=$(runpodctl get pod "$POD_ID" --output json 2>/dev/null)
POD_NAME=$(echo "$POD_INFO" | jq -r '.name // "unknown"')
VOLUME_ID=$(echo "$POD_INFO" | jq -r '.volumeId // "none"')

info "Pod Name: $POD_NAME"
info "Volume ID: $VOLUME_ID"
info "Current Status: $POD_STATUS"

# Confirm before stopping (optional safety check)
echo ""
read -p "Stop pod $POD_ID? This will halt GPU billing but preserve data on volume $VOLUME_ID. [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    info "Pod stop cancelled."
    exit 0
fi

# Stop the pod
echo -e "\n🛑 Stopping pod..."
STOP_OUTPUT=$(runpodctl stop pod "$POD_ID" 2>&1) || error "Failed to stop pod: $STOP_OUTPUT"

success "Stop command executed"

# Wait for pod to reach TERMINATED state
echo -e "\n⏳ Waiting for pod to terminate..."
STOP_TIMEOUT=120  # 2 minutes
STOP_COUNT=0

while [[ $STOP_COUNT -lt $STOP_TIMEOUT ]]; do
    CURRENT_STATUS=$(runpodctl get pod "$POD_ID" --output json 2>/dev/null | jq -r '.status' || echo "UNKNOWN")
    
    if [[ "$CURRENT_STATUS" == "TERMINATED" ]]; then
        success "Pod successfully terminated"
        break
    fi
    
    echo -n "."
    sleep 5
    STOP_COUNT=$((STOP_COUNT + 5))
done

if [[ $STOP_COUNT -ge $STOP_TIMEOUT ]]; then
    warn "Pod stop timeout. Check status manually with: runpodctl get pod $POD_ID"
else
    # Final status check
    FINAL_STATUS=$(runpodctl get pod "$POD_ID" --output json 2>/dev/null | jq -r '.status' || echo "UNKNOWN")
    
    echo -e "\n✅ Lab Environment Stopped"
    echo "=========================="
    echo ""
    echo "Pod Status: $FINAL_STATUS"
    echo "Pod ID: $POD_ID"
    echo "Volume ID: $VOLUME_ID"
    echo ""
    echo "💰 GPU billing has been halted"
    echo "💾 All data on volume $VOLUME_ID is preserved"
    echo "🔄 Run ./start-lab.sh to restart the environment"
    echo ""
    echo "Volume contents persist and will be available when you restart."
    echo "Your research data, models, and pipeline artifacts are safe."
fi

# Clean up environment variables (optional)
if [[ -f ".env" ]]; then
    # Remove pod-specific variables but keep volume ID
    grep -v "POD_ID\|SSH_HOST\|SSH_PORT" .env > .env.tmp && mv .env.tmp .env || true
fi

success "Lab shutdown complete. Volume data preserved."