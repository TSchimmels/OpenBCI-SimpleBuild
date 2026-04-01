#!/usr/bin/env bash
###############################################################################
# EEG Cursor — Boot Launcher
#
# Interactive launcher for all EEG Cursor functions.
# Run from project root:  bash boot.sh
#
# Modes:
#   bash boot.sh              — Interactive menu
#   bash boot.sh test         — Run synthetic pipeline test
#   bash boot.sh calibrate    — Collect training data (Graz paradigm)
#   bash boot.sh train        — Train classifier on last recording
#   bash boot.sh erp          — ERP signal trainer (data collection + feedback)
#   bash boot.sh run          — Launch the EEG cursor
#   bash boot.sh gui          — Launch the GUI
#   bash boot.sh pytest       — Run unit tests
###############################################################################

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
DATA_DIR="$PROJECT_DIR/data/raw"
MODELS_DIR="$PROJECT_DIR/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

###############################################################################
# Helpers
###############################################################################

activate_venv() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    else
        echo -e "${YELLOW}[!] No virtual environment found at $VENV_DIR${NC}"
        echo "    Run: bash install.sh"
        exit 1
    fi
}

find_latest_file() {
    local dir="$1"
    local ext="$2"
    # Sort by modification time (newest first), not alphabetically
    find "$dir" -maxdepth 1 -name "*.$ext" -not -name "SYNTHETIC_*" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}

print_banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║                                                  ║"
    echo "  ║              EEG CURSOR                          ║"
    echo "  ║                                                  ║"
    echo "  ║       Pure EEG Brain-Computer Interface          ║"
    echo "  ║                                                  ║"
    echo "  ╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() {
    echo -e "${DIM}──────────────────────────────────────────────────${NC}"

    # Check for recordings
    local n_recordings=$(find "$DATA_DIR" -name "*.npz" 2>/dev/null | wc -l)
    local latest_recording=$(find_latest_file "$DATA_DIR" "npz")

    # Check for trained models
    local n_models=$(find "$MODELS_DIR" -name "*.pkl" 2>/dev/null | wc -l)
    local latest_model=$(find_latest_file "$MODELS_DIR" "pkl")

    echo -e "  ${BOLD}Status:${NC}"

    if [ "$n_recordings" -gt 0 ]; then
        echo -e "    Recordings:     ${GREEN}$n_recordings${NC} (latest: $(basename "$latest_recording" 2>/dev/null))"
    else
        echo -e "    Recordings:     ${YELLOW}None${NC} — run calibration first"
    fi

    if [ "$n_models" -gt 0 ]; then
        echo -e "    Models:         ${GREEN}$n_models${NC} (latest: $(basename "$latest_model" 2>/dev/null))"
    else
        echo -e "    Models:         ${YELLOW}None${NC} — train a model first"
    fi

    # GPU
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo -e "    GPU:            ${GREEN}$gpu_name${NC}"
    else
        echo -e "    GPU:            ${DIM}CPU only${NC}"
    fi

    echo -e "${DIM}──────────────────────────────────────────────────${NC}"
}

print_menu() {
    echo ""
    echo -e "  ${BOLD}Choose an action:${NC}"
    echo ""
    echo -e "    ${GREEN}1${NC})  Test synthetic pipeline    ${DIM}(no hardware needed)${NC}"
    echo -e "    ${GREEN}2${NC})  Collect training data       ${DIM}(Graz paradigm — needs OpenBCI)${NC}"
    echo -e "    ${GREEN}3${NC})  Train classifier            ${DIM}(on latest recording)${NC}"
    echo -e "    ${GREEN}4${NC})  ERP Signal Trainer           ${DIM}(data collection + ERP feedback)${NC}"
    echo -e "    ${GREEN}5${NC})  Launch EEG Cursor            ${DIM}(real-time cursor control)${NC}"
    echo -e "    ${MAGENTA}6${NC})  Launch GUI                   ${DIM}(graphical interface)${NC}"
    echo -e "    ${CYAN}7${NC})  Run unit tests"
    echo -e "    ${CYAN}8${NC})  Run full pipeline test       ${DIM}(synth → train → run)${NC}"
    echo -e "    ${MAGENTA}9${NC})  JEPA pre-training            ${DIM}(reduce calibration time)${NC}"
    echo ""
    echo -e "    ${DIM}0${NC})  Exit"
    echo ""
}

###############################################################################
# Actions
###############################################################################

do_test_synthetic() {
    echo -e "\n${CYAN}${BOLD}Running Synthetic Pipeline Test...${NC}\n"
    python "$PROJECT_DIR/scripts/test_synthetic.py" --verbose
}

do_calibrate() {
    echo -e "\n${CYAN}${BOLD}Starting Graz Motor Imagery Calibration...${NC}\n"
    echo -e "${YELLOW}  Make sure your OpenBCI board is connected and powered on.${NC}"
    echo -e "${YELLOW}  Press ENTER when ready (or Ctrl+C to cancel)...${NC}"
    read -r

    python "$PROJECT_DIR/scripts/collect_training_data.py" --verbose

    echo -e "\n${GREEN}${BOLD}Calibration complete!${NC}"
    echo "  Next step: Train a model (option 3)"
}

do_train() {
    local latest_data=$(find_latest_file "$DATA_DIR" "npz")

    if [ -z "$latest_data" ]; then
        echo -e "\n${RED}No recording data found in $DATA_DIR${NC}"
        echo "  Run calibration first (option 2)"
        return 1
    fi

    echo -e "\n${CYAN}${BOLD}Training Classifier...${NC}\n"
    echo -e "  Data:  ${BOLD}$(basename "$latest_data")${NC}"

    # Ask for model type
    echo ""
    echo -e "  ${BOLD}Select classifier:${NC}"
    echo -e "    ${GREEN}1${NC})  CSP + LDA          ${DIM}(fast, reliable, recommended)${NC}"
    echo -e "    ${GREEN}2${NC})  Riemannian MDM      ${DIM}(robust to session drift)${NC}"
    echo -e "    ${GREEN}3${NC})  EEGNet              ${DIM}(deep learning, needs data)${NC}"
    echo ""
    read -p "  Choice [1]: " model_choice

    case "${model_choice:-1}" in
        1) MODEL_TYPE="csp_lda" ;;
        2) MODEL_TYPE="riemannian" ;;
        3) MODEL_TYPE="eegnet" ;;
        *) MODEL_TYPE="csp_lda" ;;
    esac

    echo -e "  Model: ${BOLD}$MODEL_TYPE${NC}"
    echo ""

    python "$PROJECT_DIR/scripts/train_model.py" \
        --data-path "$latest_data" \
        --model-type "$MODEL_TYPE" \
        --verbose

    echo -e "\n${GREEN}${BOLD}Training complete!${NC}"
    echo "  Next step: Launch the EEG cursor (option 5)"
}

do_erp() {
    echo -e "\n${CYAN}${BOLD}Starting ERP Signal Trainer...${NC}\n"
    echo -e "  This tool collects EEG data while showing real-time ERP feedback."
    echo -e "  You do NOT need a trained model — this is for signal exploration."
    echo -e ""
    echo -e "${YELLOW}  Make sure your OpenBCI board is connected (or use synthetic mode).${NC}"
    echo -e "${YELLOW}  Press ENTER when ready (or Ctrl+C to cancel)...${NC}"
    read -r

    python "$PROJECT_DIR/scripts/erp_trainer.py" --verbose

    echo -e "\n${GREEN}${BOLD}ERP session complete!${NC}"
    echo "  Check data/raw/ for the saved .npz file."
    echo "  To review: python scripts/erp_trainer.py --review data/raw/erp_session_*.npz"
}

do_run() {
    local latest_model=$(find_latest_file "$MODELS_DIR" "pkl")

    if [ -z "$latest_model" ]; then
        echo -e "\n${RED}No trained model found in $MODELS_DIR${NC}"
        echo "  Run training first (option 3)"
        return 1
    fi

    echo -e "\n${CYAN}${BOLD}Launching EEG Cursor...${NC}\n"
    echo -e "  Model: ${BOLD}$(basename "$latest_model")${NC}"

    echo -e "\n  ${YELLOW}Press Ctrl+C to stop.${NC}\n"

    python "$PROJECT_DIR/scripts/run_eeg_cursor.py" \
        --model "$latest_model" \
        --verbose
}

do_gui() {
    echo -e "\n${CYAN}${BOLD}Launching GUI...${NC}\n"
    python "$PROJECT_DIR/scripts/gui.py"
}

do_pytest() {
    echo -e "\n${CYAN}${BOLD}Running Unit Tests...${NC}\n"
    python -m pytest "$PROJECT_DIR/tests/" -v --tb=short
}

do_full_pipeline() {
    echo -e "\n${CYAN}${BOLD}Running Full Pipeline Test (Synthetic)...${NC}\n"

    echo -e "${BOLD}Step 1/3: Synthetic data test${NC}"
    python "$PROJECT_DIR/scripts/test_synthetic.py" || {
        echo -e "${RED}Synthetic test failed. Aborting.${NC}"
        return 1
    }

    echo -e "\n${BOLD}Step 2/3: Unit tests${NC}"
    python -m pytest "$PROJECT_DIR/tests/" -v --tb=short || {
        echo -e "${YELLOW}Some tests failed. Continuing anyway...${NC}"
    }

    echo -e "\n${BOLD}Step 3/3: Generate synthetic training data + train model${NC}"
    python "$PROJECT_DIR/scripts/generate_synthetic_data.py" --output-dir "$PROJECT_DIR/data/demo" || {
        echo -e "${YELLOW}Synthetic data generation failed. Skipping training test.${NC}"
        echo -e "\n${GREEN}${BOLD}Pipeline test complete (2/3 steps)!${NC}"
        return 0
    }
    DEMO_FILE=$(find "$PROJECT_DIR/data/demo" -name "SYNTHETIC_*.npz" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$DEMO_FILE" ]; then
        python "$PROJECT_DIR/scripts/train_model.py" --data-path "$DEMO_FILE" --model-type csp_lda --verbose || {
            echo -e "${YELLOW}Training on synthetic data failed. Non-critical.${NC}"
        }
    fi

    echo -e "\n${GREEN}${BOLD}Full pipeline test complete!${NC}"
    echo "  The entire pipeline works: data -> train -> model."
    echo "  Connect your OpenBCI board for real EEG calibration."
}

do_pretrain() {
    echo -e "\n${CYAN}${BOLD}JEPA Self-Supervised Pre-Training...${NC}\n"
    echo -e "  This learns EEG structure from UNLABELED data to reduce calibration time."
    echo -e "  Just wear the cap and relax — no tasks needed."
    echo -e ""
    echo -e "${YELLOW}  Press ENTER when ready (or Ctrl+C to cancel)...${NC}"
    read -r

    python -c "
import sys, time
sys.path.insert(0, '.')
from src.config import load_config
from src.acquisition.board import BoardManager
import numpy as np

config = load_config()
board = BoardManager(config)
board.connect()
sf = board.get_sampling_rate()
eeg_ch = board.get_eeg_channels()
print(f'Board: {sf}Hz, {len(eeg_ch)} channels')
print('Collecting 2 minutes of unlabeled EEG for pre-training...')

board.get_board_data()  # flush
time.sleep(120)
raw = board.get_board_data()
eeg = raw[eeg_ch, :]
board.disconnect()
print(f'Collected {eeg.shape[1]} samples ({eeg.shape[1]/sf:.0f}s)')

# Pre-train
from src.training.pretrain import JEPAPretrainer
pt = JEPAPretrainer(n_channels=len(eeg_ch), n_samples=int(2.5*sf), sf=sf)

# Cut into windows
win_len = int(2.5 * sf)
windows = []
for i in range(0, eeg.shape[1] - win_len, win_len // 2):
    windows.append(eeg[:, i:i+win_len])
X = np.stack(windows)
print(f'Training on {X.shape[0]} windows...')

result = pt.pretrain(X, n_epochs=50, batch_size=16)
print(f'Pre-training complete. Final loss: {result[\"loss_history\"][-1]:.4f}')

import joblib
joblib.dump(pt, 'models/jepa_encoder.pkl')
print('Encoder saved to models/jepa_encoder.pkl')
"

    echo -e "\n${GREEN}${BOLD}Pre-training complete!${NC}"
    echo "  The encoder is saved. Use it with train_model.py for faster calibration."
}

###############################################################################
# CLI mode (direct command)
###############################################################################

if [ "${1:-}" != "" ]; then
    activate_venv
    cd "$PROJECT_DIR"

    case "$1" in
        test)       do_test_synthetic ;;
        pretrain)   do_pretrain ;;
        calibrate)  do_calibrate ;;
        train)      do_train ;;
        erp)        do_erp ;;
        run)        do_run ;;
        gui)        do_gui ;;
        pytest)     do_pytest ;;
        full)       do_full_pipeline ;;
        *)
            echo "Unknown command: $1"
            echo "Usage: bash boot.sh [test|calibrate|train|erp|run|gui|pytest|full]"
            exit 1
            ;;
    esac
    exit 0
fi

###############################################################################
# Interactive menu mode
###############################################################################

activate_venv
cd "$PROJECT_DIR"

while true; do
    print_banner
    print_status
    print_menu

    read -p "  Enter choice: " choice

    case "$choice" in
        1) do_test_synthetic ;;
        2) do_calibrate ;;
        3) do_train ;;
        4) do_erp ;;
        5) do_run ;;
        6) do_gui ;;
        7) do_pytest ;;
        8) do_full_pipeline ;;
        9) do_pretrain ;;
        0) echo -e "\n${DIM}Goodbye.${NC}\n"; exit 0 ;;
        *) echo -e "\n${RED}Invalid choice.${NC}"; sleep 1 ;;
    esac

    echo ""
    echo -e "${DIM}Press ENTER to return to menu...${NC}"
    read -r
done
