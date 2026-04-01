#!/usr/bin/env bash
###############################################################################
# EEG Cursor — Installer
#
# Installs all dependencies and validates the environment.
# Run from the project root:  bash install.sh
#
# Supports: Ubuntu/Debian, Fedora/RHEL/CentOS, Arch/Manjaro, macOS, WSL2
# GPU:      NVIDIA CUDA (auto-detected), Apple MPS (auto-detected)
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

# ---- OS / distro detection ----
OS_TYPE="$(uname -s)"
ARCH="$(uname -m)"
DISTRO="unknown"
PKG_MANAGER="none"
IS_WSL=false

if [[ "$OS_TYPE" == "Darwin" ]]; then
    DISTRO="macOS"
    PKG_MANAGER="brew"
elif [[ "$OS_TYPE" == "Linux" ]]; then
    # Detect WSL
    if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
        IS_WSL=true
    fi
    # Detect distro
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian|pop|linuxmint|elementary|zorin)
                DISTRO="$ID"
                PKG_MANAGER="apt"
                ;;
            fedora|rhel|centos|rocky|alma|nobara)
                DISTRO="$ID"
                PKG_MANAGER="dnf"
                # RHEL/CentOS 7 uses yum
                command -v dnf &>/dev/null || PKG_MANAGER="yum"
                ;;
            arch|manjaro|endeavouros|garuda)
                DISTRO="$ID"
                PKG_MANAGER="pacman"
                ;;
            opensuse*|sles)
                DISTRO="$ID"
                PKG_MANAGER="zypper"
                ;;
            *)
                DISTRO="$ID"
                # Try to detect package manager
                if command -v apt-get &>/dev/null; then PKG_MANAGER="apt"
                elif command -v dnf &>/dev/null; then PKG_MANAGER="dnf"
                elif command -v yum &>/dev/null; then PKG_MANAGER="yum"
                elif command -v pacman &>/dev/null; then PKG_MANAGER="pacman"
                elif command -v zypper &>/dev/null; then PKG_MANAGER="zypper"
                fi
                ;;
        esac
    fi
fi

echo -e "  Project:  ${BOLD}$PROJECT_DIR${NC}"
echo -e "  Python:   ${BOLD}$($PYTHON --version 2>&1)${NC}"
echo -e "  Platform: ${BOLD}$OS_TYPE $ARCH${NC}"
echo -e "  Distro:   ${BOLD}$DISTRO${NC}"
echo -e "  Pkg Mgr:  ${BOLD}$PKG_MANAGER${NC}"
if [ "$IS_WSL" = true ]; then
    echo -e "  WSL:      ${BOLD}yes${NC}"
fi
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

# Check python3-venv is available
if ! $PYTHON -m venv --help &>/dev/null; then
    print_error "python3-venv is required but not installed."
    case "$PKG_MANAGER" in
        apt)    echo "  Install with: sudo apt install python3-venv" ;;
        dnf)    echo "  Install with: sudo dnf install python3-devel" ;;
        pacman) echo "  Install with: sudo pacman -S python" ;;
        brew)   echo "  Python from Homebrew includes venv by default" ;;
    esac
    exit 1
fi

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

GPU_TYPE="none"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        print_ok "NVIDIA GPU: $GPU_NAME ($GPU_MEM)"
        GPU_TYPE="cuda"
    fi
elif [[ "$OS_TYPE" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
    print_ok "Apple Silicon detected (MPS acceleration available)"
    GPU_TYPE="mps"
fi

if [ "$GPU_TYPE" = "none" ]; then
    print_warn "No GPU detected. EEGNet will use CPU (slower but works)."
fi

###############################################################################
# 4. Install PyTorch (GPU-aware)
###############################################################################
print_header "Installing PyTorch"

case "$GPU_TYPE" in
    cuda)
        print_step "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
        print_ok "PyTorch installed (CUDA)"
        ;;
    mps)
        print_step "Installing PyTorch with MPS support (Apple Silicon)..."
        pip install torch torchvision torchaudio -q
        print_ok "PyTorch installed (MPS)"
        ;;
    *)
        print_step "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
        print_ok "PyTorch installed (CPU)"
        ;;
esac

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
# 6. System dependencies (OS-specific)
###############################################################################
print_header "System Dependencies"

install_sys_deps() {
    case "$PKG_MANAGER" in
        apt)
            print_step "Installing system dependencies (apt)..."
            local pkgs="libsdl2-mixer-2.0-0 libsdl2-2.0-0 libsdl2-dev xdotool xsel xclip"
            sudo apt-get install -y $pkgs 2>/dev/null || \
                print_warn "Some system packages failed to install (may need manual install)"
            print_ok "apt packages installed"
            ;;
        dnf|yum)
            print_step "Installing system dependencies ($PKG_MANAGER)..."
            local pkgs="SDL2 SDL2-devel SDL2_mixer xdotool xsel xclip python3-tkinter"
            sudo $PKG_MANAGER install -y $pkgs 2>/dev/null || \
                print_warn "Some system packages failed to install (may need manual install)"
            print_ok "$PKG_MANAGER packages installed"
            ;;
        pacman)
            print_step "Installing system dependencies (pacman)..."
            local pkgs="sdl2 sdl2_mixer xdotool xsel xclip tk"
            sudo pacman -S --noconfirm --needed $pkgs 2>/dev/null || \
                print_warn "Some system packages failed to install (may need manual install)"
            print_ok "pacman packages installed"
            ;;
        zypper)
            print_step "Installing system dependencies (zypper)..."
            local pkgs="libSDL2-2_0-0 libSDL2_mixer-2_0-0 xdotool xsel xclip"
            sudo zypper install -y $pkgs 2>/dev/null || \
                print_warn "Some system packages failed to install (may need manual install)"
            print_ok "zypper packages installed"
            ;;
        brew)
            print_step "Installing system dependencies (Homebrew)..."
            brew install sdl2 sdl2_mixer 2>/dev/null || \
                print_warn "Homebrew packages failed. Install SDL2 manually."
            # macOS doesn't need xdotool (PyAutoGUI uses native APIs)
            print_ok "Homebrew packages installed"
            ;;
        *)
            print_warn "Unknown package manager. You may need to install SDL2 and xdotool manually."
            print_warn "  SDL2 — required for pygame audio (training paradigm beep)"
            print_warn "  xdotool — required for PyAutoGUI on Linux (cursor control)"
            ;;
    esac
}

install_sys_deps

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
if [ "$GPU_TYPE" = "cuda" ]; then
    CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAIL" = "True" ]; then
        CUDA_DEV=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_ok "CUDA available: $CUDA_DEV"
    else
        print_warn "CUDA not available to PyTorch (driver mismatch?)"
    fi
elif [ "$GPU_TYPE" = "mps" ]; then
    MPS_AVAIL=$(python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null)
    if [ "$MPS_AVAIL" = "True" ]; then
        print_ok "MPS available (Apple Silicon GPU acceleration)"
    else
        print_warn "MPS not available to PyTorch (macOS 12.3+ required)"
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
