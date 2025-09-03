#!/bin/bash
set -euo pipefail

# Genesis: Bootstrap script for Training LLMs to Manipulate Cogit Hypervectors research environment
# Defensive research using synthetic data only

echo "🧬 Genesis: Initializing Cogit Hypervector Research Environment"
echo "============================================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() { echo -e "${RED}ERROR: $1${NC}" >&2; exit 1; }
warn() { echo -e "${YELLOW}WARNING: $1${NC}" >&2; }
success() { echo -e "${GREEN}✓ $1${NC}"; }
info() { echo "ℹ️  $1"; }

[[ "$OSTYPE" == "darwin"* ]] || error "This script is designed for macOS"

# 1. Verify/Install Dependencies with idempotency checks
echo -e "\n📦 Checking Dependencies..."

# Homebrew
if ! command -v brew &> /dev/null; then
    warn "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    success "Homebrew installed"
else
    success "Homebrew found"
fi

brew update &> /dev/null || warn "Failed to update Homebrew"

# Git, jq - provide brew commands where appropriate
for pkg in git jq; do
    if ! command -v "$pkg" &> /dev/null; then
        info "Installing $pkg via brew install $pkg..."
        brew install "$pkg" || error "Failed to install $pkg"
        success "$pkg installed"
    else
        success "$pkg found"
    fi
done

# DVC
if ! command -v dvc &> /dev/null; then
    info "Installing DVC via brew install dvc..."
    brew install dvc || error "Failed to install DVC"
    success "DVC installed"
else
    success "DVC found"
fi

# runpodctl - use wget commands as fallback
if ! command -v runpodctl &> /dev/null; then
    info "Installing runpodctl..."
    RUNPOD_URL=$(curl -s https://api.github.com/repos/runpod/runpodctl/releases/latest | \
        jq -r '.assets[] | select(.name | contains("darwin") and contains("amd64")) | .browser_download_url')
    
    if [[ -z "$RUNPOD_URL" ]]; then
        error "Failed to find runpodctl download URL. Try: wget <URL> manually"
    fi
    
    curl -L "$RUNPOD_URL" -o /tmp/runpodctl.tar.gz || error "Download failed. Try: wget $RUNPOD_URL"
    tar -xzf /tmp/runpodctl.tar.gz -C /tmp
    sudo mv /tmp/runpodctl /usr/local/bin/
    chmod +x /usr/local/bin/runpodctl
    rm -f /tmp/runpodctl.tar.gz
    success "runpodctl installed"
else
    success "runpodctl found"
fi

# 2. Configure RunPod API - confirm by listing config without printing secrets
echo -e "\n🔑 RunPod Configuration..."

if runpodctl config --help &> /dev/null && runpodctl get pods &> /dev/null 2>&1; then
    success "RunPod API already configured"
    info "Current config confirmed (pods accessible)"
else
    echo "Enter your RunPod API key:"
    read -s RUNPOD_API_KEY
    [[ -n "$RUNPOD_API_KEY" ]] || error "API key cannot be empty"
    
    runpodctl config --apiKey "$RUNPOD_API_KEY" || error "Failed to configure RunPod API key"
    
    if runpodctl get pods &> /dev/null; then
        success "RunPod API configured and verified"
    else
        error "RunPod API configuration failed - check your key"
    fi
fi

# 3. SSH Key Management - ensure ed25519 exists, print public key  
echo -e "\n🔐 SSH Key Setup..."

SSH_KEY_PATH="$HOME/.ssh/id_ed25519"

if [[ ! -f "$SSH_KEY_PATH" ]]; then
    warn "Generating ed25519 SSH key..."
    ssh-keygen -t ed25519 -C "genesis-cogit-research" -f "$SSH_KEY_PATH" -N ""
    success "SSH key generated"
else
    success "SSH key found"
fi

echo -e "\n📋 Your SSH public key (register in RunPod settings):"
echo "=================================================="
cat "${SSH_KEY_PATH}.pub"
echo "=================================================="
echo "1. Copy the above SSH public key"
echo "2. Go to https://www.runpod.io/console/serverless/user/settings"  
echo "3. Add the key to your SSH Keys section"
echo "4. Press ENTER when done..."
read -r

# 4. Create Network Volume - parse runpodctl create --help to discover correct resource noun
echo -e "\n💾 Creating Network Volume..."

# Runtime CLI discovery - avoid hard-coded subcommand guesses
VOLUME_HELP=$(runpodctl create --help 2>&1 || echo "")
if echo "$VOLUME_HELP" | grep -q "network-volume"; then
    VOLUME_RESOURCE="network-volume"
elif echo "$VOLUME_HELP" | grep -q "volume"; then
    VOLUME_RESOURCE="volume"  
else
    error "Cannot determine network volume resource name from: runpodctl create --help"
fi

info "Using discovered resource name: $VOLUME_RESOURCE"

# Check existing volumes
EXISTING_VOLUMES=$(runpodctl get "$VOLUME_RESOURCE" 2>/dev/null | tail -n +2 || echo "")
COGIT_VOLUME_ID=""

if echo "$EXISTING_VOLUMES" | grep -q "cogit-research"; then
    COGIT_VOLUME_ID=$(echo "$EXISTING_VOLUMES" | grep "cogit-research" | awk '{print $1}' | head -n1)
    success "Found existing volume: $COGIT_VOLUME_ID"
else
    info "Creating 500GB network volume in chosen region..."
    CREATE_CMD="runpodctl create $VOLUME_RESOURCE --name cogit-research --size 500 --dataCenterId US-OR-1"
    info "Executing command: $CREATE_CMD"
    
    CREATE_OUTPUT=$($CREATE_CMD 2>&1) || error "Failed to create network volume: $CREATE_OUTPUT"
    
    # Extract volume ID for later use
    COGIT_VOLUME_ID=$(echo "$CREATE_OUTPUT" | grep -oE '[a-z0-9]{24}' | head -n1)
    [[ -n "$COGIT_VOLUME_ID" ]] || error "Could not extract volume ID from output: $CREATE_OUTPUT"
    
    success "Network volume created: $COGIT_VOLUME_ID"
fi

echo "export COGIT_VOLUME_ID=\"$COGIT_VOLUME_ID\"" > .env
success "Volume ID saved to .env for later use"

# 5. Initialize Repository - project-cogit-framework with Python tooling
echo -e "\n📁 Creating Project Repository..."

PROJECT_DIR="project-cogit-framework"

if [[ -d "$PROJECT_DIR" ]]; then
    warn "Using existing project directory"
    cd "$PROJECT_DIR"
else
    mkdir "$PROJECT_DIR" && cd "$PROJECT_DIR"
    success "Project directory created"
fi

# Git initialization
if [[ ! -d ".git" ]]; then
    git init
    success "Git repository initialized"
fi

# Python project structure - pyproject.toml with ruff/black/pytest
if [[ ! -f "pyproject.toml" ]]; then
    cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cogit-framework"
version = "0.1.0"
description = "Training LLMs to Manipulate Cogit Hypervectors - Defensive Research"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
    "torchhd",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "transformers>=4.20.0",
    "datasets>=2.0.0",
    "jsonlines>=3.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.64.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "ruff>=0.1.0",
]

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I"]
EOF
    success "pyproject.toml with Python tooling created"
fi

# DVC initialization - include PYTHONHASHSEED environment variable
if [[ ! -d ".dvc" ]]; then
    export PYTHONHASHSEED=42
    dvc init --no-scm
    success "DVC initialized with deterministic seeding"
fi

# 6. First-run guidance - exact commands for DVC remote, dvc repro, experiment branches
echo -e "\n🚀 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps - Exact Commands:"
echo ""
echo "1. Copy study overview PDF to anchor 'cogit' concept:"
echo "   mkdir -p docs && cp ~/Documents/Bagley-Morrissey-study-overview.pdf docs/"
echo ""
echo "2. Start your research lab environment:"
echo "   ./start-lab.sh"
echo ""
echo "3. Connect via VS Code/Cursor Remote-SSH (instructions will be provided)"
echo ""
echo "4. Initialize DVC remote and run pipeline (all artifacts persist on /workspace):"
echo "   dvc remote add -d cogit-storage /workspace/dvc-cache"
echo "   dvc repro"
echo "   git add dvc.yaml dvc.lock .dvc/config"
echo "   git commit -m 'feat: initialize three-stage cogit pipeline'"
echo ""
echo "5. Create experiment branch for hypothesis iteration:"
echo "   git checkout -b exp/baseline-simulation-v1"
echo ""
echo "6. Tag experiment branches for ablation studies:"
echo "   git tag exp-baseline-v1 && git push origin exp-baseline-v1"
echo ""
echo "⚠️  IMPORTANT: This is defensive research using synthetic data only!"
echo "📖 Consult docs/Bagley-Morrissey-study-overview.pdf before Stage 1 prompts."
echo "📁 All artifacts live under /workspace on the pod (persisted by network volume)."
echo ""
echo "Volume ID: $COGIT_VOLUME_ID"
echo "Project directory: $(pwd)"

export COGIT_VOLUME_ID