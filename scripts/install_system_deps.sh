#!/bin/bash
# Script to install system dependencies required for Photo Derush
# These are Qt platform plugins dependencies that cannot be managed by Poetry

set -e

echo "Checking system dependencies for Photo Derush..."

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect Linux distribution. Please install manually:"
    echo "  libxcb-xinerama0 libxcb-cursor0 libxcb1 libxkbcommon-x11-0"
    exit 1
fi

case $OS in
    ubuntu|debian)
        echo "Detected Ubuntu/Debian. Installing dependencies..."
        sudo apt-get update
        sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxcb1 libxkbcommon-x11-0
        ;;
    fedora|rhel|centos)
        echo "Detected Fedora/RHEL/CentOS. Installing dependencies..."
        sudo dnf install -y libxcb xcb-util-cursor libxkbcommon-x11
        ;;
    arch|manjaro)
        echo "Detected Arch Linux. Installing dependencies..."
        sudo pacman -S --noconfirm libxcb libxcb-cursor libxkbcommon-x11
        ;;
    *)
        echo "Unsupported distribution: $OS"
        echo "Please install manually:"
        echo "  libxcb-xinerama0 libxcb-cursor0 libxcb1 libxkbcommon-x11-0"
        exit 1
        ;;
esac

echo "System dependencies installed successfully!"
echo "You can now run: poetry run python app.py"

