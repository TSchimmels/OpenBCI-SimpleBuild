#!/usr/bin/env bash
###############################################################################
# EEG Cursor — Installer
#
# Installs all dependencies and validates the environment.
# Run from the project root:  bash install.sh
#
# Supports: Ubuntu/Debian (WSL2), native Linux
# GPU:      NVIDIA CUDA (auto-detected)
###############################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="${PYTHON:-python3}"

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}============================================${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}============================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[X]${NC} $1"
}

print_ok() {
    echo -e "${GREEN}[✓]${NC} $1"
}

###############################################################################
# 1. System checks
###############################################################################
print_header "EEG Cursor — Installer"

echo -e "  Project:  ${BOLD}$PROJECT_DIR${NC}"
echo -e "  Python:   ${BOLD}$($PYTHON --version 2>&1)${NC}"
echo -e "  Platform: ${BOLD}$(uname -s) $(uname -m)${NC}"
echo ""

# Check Python version >= 3.10
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    print_error "Python 3.10+ required. Found: $PY_VERSION"
    exit 1
fi
print_ok "Python $PY_VERSION"

###############################################################################
# 2. Create virtual environment
###############################################################################
print_header "Setting Up Virtual Environment"

if [ -d "$VENV_DIR" ]; then
    print_warn "Virtual environment already exists at $VENV_DIR"
    read -p "  Recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
        print_step "Recreated virtual environment"
    else
        print_step "Using existing virtual environment"
    fi
else
    $PYTHON -m venv "$VENV_DIR"
    print_step "Created virtual environment at $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
print_ok "Activated: $(which python)"

# Upgrade pip
pip install --upgrade pip setuptools wheel -q
print_ok "pip upgraded"

###############################################################################
# 3. Detect GPU
###############################################################################
print_header "Detecting GPU"

HAS_GPU=false
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        print_ok "GPU: $GPU_NAME ($GPU_MEM)"
        HAS_GPU=true
    fi
fi

if [ "$HAS_GPU" = false ]; then
    print_warn "No NVIDIA GPU detected. EEGNet will use CPU (slower but works)."
fi

###############################################################################
# 4. Install PyTorch (GPU-aware)
###############################################################################
print_header "Installing PyTorch"

if [ "$HAS_GPU" = true ]; then
    print_step "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
    print_ok "PyTorch installed (CUDA)"
else
    print_step "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    print_ok "PyTorch installed (CPU)"
fi

###############################################################################
# 5. Install core dependencies
###############################################################################
print_header "Installing Dependencies"

# Core EEG
print_step "BrainFlow + MNE + scipy..."
pip install brainflow mne numpy scipy scikit-learn -q
print_ok "Core EEG stack"

# ML / Classification
print_step "pyRiemann + braindecode..."
pip install pyriemann braindecode -q
print_ok "ML classifiers"

# Nonlinear features
print_step "antropy (chaos features)..."
pip install antropy -q
print_ok "Nonlinear features"

# Cursor control
print_step "PyAutoGUI..."
pip install pyautogui -q
print_ok "Cursor control"

# Visualization
print_step "pyqtgraph + PyQt5 + matplotlib..."
pip install pyqtgraph PyQt5 matplotlib -q
print_ok "Visualization"

# Training paradigm
print_step "pygame..."
pip install pygame -q
print_ok "Training paradigm"

# Config + testing
print_step "PyYAML + pytest..."
pip install pyyaml pytest -q
print_ok "Config + testing"

# Benchmarking (optional)
print_step "MOABB (benchmarking)..."
pip install moabb -q 2>/dev/null || print_warn "MOABB install failed (optional, non-critical)"
print_ok "Benchmarking"

###############################################################################
# 6. System dependencies (Linux/WSL)
###############################################################################
print_header "System Dependencies"

# For pygame audio
if command -v apt-get &>/dev/null; then
    print_step "Checking system audio libraries for pygame..."
    if ! dpkg -s libsdl2-mixer-2.0-0 &>/dev/null 2>&1; then
        print_warn "Installing SDL2 mixer for pygame audio (may need sudo)..."
        sudo apt-get install -y libsdl2-mixer-2.0-0 libsdl2-2.0-0 2>/dev/null || \
            print_warn "Could not install SDL2. Paradigm beep may not work."
    fi
    print_ok "System audio libraries"
fi

# For PyAutoGUI on Linux
if [[ "$(uname -s)" == "Linux" ]]; then
    if ! dpkg -s xdotool &>/dev/null 2>&1; then
        print_warn "Installing xdotool for PyAutoGUI..."
        sudo apt-get install -y xdotool xsel 2>/dev/null || \
            print_warn "Could not install xdotool. Mouse control may not work on Linux."
    fi
fi

###############################################################################
# 7. Create directories
###############################################################################
print_header "Project Directories"

mkdir -p "$PROJECT_DIR"/{data/{raw,processed,physionet},models,notebooks}
print_ok "data/raw, data/processed, data/physionet, models, notebooks"

###############################################################################
# 8. Validate installation
###############################################################################
print_header "Validating Installation"

FAILED=0

check_import() {
    local name="$1"
    local import_name="${2:-$1}"
    if python -c "import $import_name" 2>/dev/null; then
        print_ok "$name"
    else
        print_error "$name — FAILED TO IMPORT"
        FAILED=$((FAILED + 1))
    fi
}

check_import "BrainFlow" "brainflow"
check_import "MNE-Python" "mne"
check_import "scikit-learn" "sklearn"
check_import "pyRiemann" "pyriemann"
check_import "PyTorch" "torch"
check_import "braindecode" "braindecode"
check_import "antropy" "antropy"
check_import "PyAutoGUI" "pyautogui"
check_import "pygame" "pygame"
check_import "pyqtgraph" "pyqtgraph"
check_import "PyYAML" "yaml"
check_import "pytest" "pytest"
check_import "NumPy" "numpy"
check_import "SciPy" "scipy"

echo ""

# GPU validation
if [ "$HAS_GPU" = true ]; then
    CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAIL" = "True" ]; then
        CUDA_DEV=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_ok "CUDA available: $CUDA_DEV"
    else
        print_warn "CUDA not available to PyTorch (driver mismatch?)"
    fi
fi

###############################################################################
# 9. Run quick smoke test
###############################################################################
print_header "Quick Smoke Test"

print_step "Testing BrainFlow synthetic board..."
python -c "
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
params = BrainFlowInputParams()
board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
board.prepare_session()
board.start_stream()
import time; time.sleep(1)
data = board.get_board_data()
board.stop_stream()
board.release_session()
print(f'  Channels: {data.shape[0]}, Samples: {data.shape[1]}')
assert data.shape[1] > 0, 'No data received'
print('  BrainFlow synthetic board: OK')
" 2>&1 && print_ok "BrainFlow smoke test passed" || print_error "BrainFlow smoke test FAILED"

print_step "Testing preprocessing pipeline..."
python -c "
import numpy as np
from src.preprocessing.filters import bandpass_filter, common_average_reference
data = np.random.randn(8, 250)
filtered = bandpass_filter(data, sf=250, low=8.0, high=30.0, causal=False)
car = common_average_reference(filtered)
assert car.shape == data.shape
print('  Preprocessing: OK')
" 2>&1 && print_ok "Preprocessing smoke test passed" || print_error "Preprocessing smoke test FAILED"

###############################################################################
# 10. Summary
###############################################################################
print_header "Installation Complete"

if [ "$FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}All packages installed successfully.${NC}"
else
    echo -e "  ${YELLOW}${BOLD}$FAILED package(s) failed to install.${NC}"
fi

echo ""
echo -e "  ${BOLD}To activate the environment:${NC}"
echo -e "    source $VENV_DIR/bin/activate"
echo ""
echo -e "  ${BOLD}To run tests:${NC}"
echo -e "    python -m pytest tests/ -v"
echo ""
echo -e "  ${BOLD}To test the full pipeline:${NC}"
echo -e "    python scripts/test_synthetic.py"
echo ""
echo -e "  ${BOLD}To start the EEG Cursor:${NC}"
echo -e "    bash boot.sh"
echo ""
echo -e "  ${BOLD}Or launch the GUI:${NC}"
echo -e "    python scripts/gui.py"
echo ""
