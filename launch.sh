#!/usr/bin/env bash
###############################################################################
# EEG Cursor — One-Click Launcher
#
# Usage:
#   bash launch.sh          — Launch the GUI
#   bash launch.sh --cli    — Launch the interactive menu (boot.sh)
#   bash launch.sh --test   — Run pipeline test + unit tests
#   bash launch.sh --install — Run the installer first, then launch
###############################################################################

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ------------------------------------------------------------------
# Ensure virtual environment exists
# ------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    if [ "${1:-}" = "--install" ]; then
        echo -e "${CYAN}${BOLD}Running installer...${NC}"
        bash "$PROJECT_DIR/install.sh"
    else
        echo -e "${RED}No virtual environment found.${NC}"
        echo -e "Run:  ${BOLD}bash launch.sh --install${NC}"
        exit 1
    fi
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

# ------------------------------------------------------------------
# Route command
# ------------------------------------------------------------------
case "${1:-}" in
    --cli)
        exec bash "$PROJECT_DIR/boot.sh"
        ;;
    --test)
        echo -e "${CYAN}${BOLD}Running tests...${NC}\n"
        python scripts/test_synthetic.py --verbose
        echo ""
        python -m pytest tests/ -v --tb=short
        echo -e "\n${GREEN}${BOLD}All tests passed.${NC}"
        ;;
    --install)
        echo -e "\n${GREEN}${BOLD}Install complete. Launching GUI...${NC}\n"
        exec python scripts/gui.py
        ;;
    ""|--gui)
        exec python scripts/gui.py
        ;;
    *)
        echo "Usage: bash launch.sh [--gui|--cli|--test|--install]"
        exit 1
        ;;
esac
