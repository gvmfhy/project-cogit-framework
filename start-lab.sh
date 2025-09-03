#!/bin/bash
set -euo pipefail

# start-lab.sh: Find or create RTX-class GPU pod, attach network volume, verify environment
# All execution, storage, and GPU compute occur on RunPod with /workspace persistence

echo "🚀 Starting Cogit Research Lab Environment"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() { echo -e "${RED}ERROR: $1${NC}" >&2; exit 1; }
warn() { echo -e "${YELLOW}WARNING: $1${NC}" >&2; }
success() { echo -e "${GREEN}✓ $1${NC}"; }
info() { echo "ℹ️  $1"; }

# Load volume ID from environment
if [[ -f ".env" ]]; then
    source .env
fi

[[ -n "${COGIT_VOLUME_ID:-}" ]] || error "COGIT_VOLUME_ID not found. Run genesis.sh first."

# Check for existing pods
echo -e "\n🔍 Checking for existing pods..."
EXISTING_PODS=$(runpodctl get pods --output json 2>/dev/null || echo "[]")

# Look for running pod with our volume
RUNNING_POD=""
if [[ "$EXISTING_PODS" != "[]" ]]; then
    # Parse JSON to find pod with our volume (simplified check)
    RUNNING_POD=$(echo "$EXISTING_PODS" | jq -r '.[] | select(.status == "RUNNING") | .id' | head -n1)
fi

if [[ -n "$RUNNING_POD" ]]; then
    info "Found existing running pod: $RUNNING_POD"
    POD_ID="$RUNNING_POD"
else
    echo -e "\n🖥️  Creating new RTX-class GPU pod..."
    
    # Create pod suitable for PyTorch with network volume
    CREATE_CMD="runpodctl create pod \
        --name cogit-research-lab \
        --imageName runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 \
        --gpuType 'RTX 4090' \
        --volumeId $COGIT_VOLUME_ID \
        --volumeMountPath /workspace \
        --containerDiskInGb 50 \
        --minMemoryInGb 16 \
        --minVcpuCount 8 \
        --env PYTHONHASHSEED=42"
    
    info "Executing: $CREATE_CMD"
    
    CREATE_OUTPUT=$(eval "$CREATE_CMD" 2>&1) || error "Failed to create pod: $CREATE_OUTPUT"
    
    # Extract pod ID from output
    POD_ID=$(echo "$CREATE_OUTPUT" | grep -oE '[a-z0-9]{24}' | head -n1)
    [[ -n "$POD_ID" ]] || error "Could not extract pod ID from: $CREATE_OUTPUT"
    
    success "Pod created: $POD_ID"
fi

# Wait for pod to be ready
echo -e "\n⏳ Waiting for pod to be ready..."
READY_TIMEOUT=300  # 5 minutes
READY_COUNT=0

while [[ $READY_COUNT -lt $READY_TIMEOUT ]]; do
    POD_STATUS=$(runpodctl get pod "$POD_ID" --output json | jq -r '.status' 2>/dev/null || echo "UNKNOWN")
    
    if [[ "$POD_STATUS" == "RUNNING" ]]; then
        success "Pod is running"
        break
    elif [[ "$POD_STATUS" == "FAILED" ]]; then
        error "Pod failed to start"
    fi
    
    echo -n "."
    sleep 5
    READY_COUNT=$((READY_COUNT + 5))
done

[[ $READY_COUNT -lt $READY_TIMEOUT ]] || error "Pod startup timeout"

# Critical: Assert /workspace exists and is writable
echo -e "\n🔍 Verifying /workspace mount..."

if ! runpodctl ssh "$POD_ID" "test -d /workspace" 2>/dev/null; then
    error "/workspace directory does not exist. Network volume not properly mounted. Check volume attachment."
fi

if ! runpodctl ssh "$POD_ID" "touch /workspace/.genesis_probe && rm -f /workspace/.genesis_probe" 2>/dev/null; then
    error "/workspace is not writable. Check volume permissions and mount status."
fi

success "/workspace verified as mounted and writable"

# Verify nvidia-smi and print GPU information
echo -e "\n🎮 Verifying GPU access..."

GPU_INFO=$(runpodctl ssh "$POD_ID" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits" 2>/dev/null) || {
    error "nvidia-smi failed. GPU not available or drivers not properly installed."
}

if [[ -n "$GPU_INFO" ]]; then
    echo "GPU Information:"
    echo "$GPU_INFO" | while read -r line; do
        echo "  📊 $line"
    done
    success "GPU verified and accessible"
else
    error "No GPU information returned"
fi

# Get SSH connection details
echo -e "\n🔗 Getting SSH connection details..."

SSH_INFO=$(runpodctl get pod "$POD_ID" --output json | jq -r '.runtime.ports[] | select(.privatePort == 22)')
SSH_HOST=$(echo "$SSH_INFO" | jq -r '.ip')
SSH_PORT=$(echo "$SSH_INFO" | jq -r '.publicPort')

[[ "$SSH_HOST" != "null" && "$SSH_PORT" != "null" ]] || error "Could not retrieve SSH connection details"

# Setup project on pod
echo -e "\n📁 Setting up project on pod..."

runpodctl ssh "$POD_ID" "mkdir -p /workspace/project-cogit-framework" 2>/dev/null || warn "Project directory may already exist"

# Copy project files to pod (if not already synced)
info "Project files can be synced via Remote-SSH or using rsync"

# Print connection instructions
echo -e "\n🎯 Lab Environment Ready!"
echo "========================="
echo ""
echo "SSH Connection Details:"
echo "  Host: $SSH_HOST"
echo "  Port: $SSH_PORT" 
echo ""
echo "VS Code/Cursor Remote-SSH Configuration:"
echo "  1. Open VS Code/Cursor"
echo "  2. Install 'Remote - SSH' extension if not already installed"
echo "  3. Press Cmd+Shift+P (Mac) or Ctrl+Shift+P (Windows/Linux)"
echo "  4. Type 'Remote-SSH: Connect to Host'"
echo "  5. Enter: root@$SSH_HOST -p $SSH_PORT"
echo "  6. When connected, open folder: /workspace/project-cogit-framework"
echo ""
echo "Direct SSH (from terminal):"
echo "  ssh root@$SSH_HOST -p $SSH_PORT"
echo ""
echo "Important Notes:"
echo "  🔒 All data in /workspace persists when pod stops"
echo "  💾 Volume ID: $COGIT_VOLUME_ID"
echo "  🆔 Pod ID: $POD_ID"
echo "  🌍 Environment: PYTHONHASHSEED=42 is set"
echo ""
echo "Next steps:"
echo "  1. Connect via Remote-SSH"
echo "  2. cd /workspace/project-cogit-framework"
echo "  3. Set up DVC remote: dvc remote add -d cogit-storage /workspace/dvc-cache"
echo "  4. Run pipeline: dvc repro"
echo ""

# Save connection info for future reference
echo "export POD_ID=\"$POD_ID\"" >> .env
echo "export SSH_HOST=\"$SSH_HOST\"" >> .env
echo "export SSH_PORT=\"$SSH_PORT\"" >> .env

success "Lab environment ready! Connect via Remote-SSH to begin research."