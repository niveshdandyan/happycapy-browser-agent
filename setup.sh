#!/bin/bash
# Browser-Use Setup Script
# Installs browser-use and Chromium for AI browser automation

set -e

echo "🌐 Browser-Use Setup"
echo "===================="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Python $PYTHON_VERSION detected"

# Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo ""
    echo "🔨 Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Install browser-use
echo ""
echo "📦 Installing browser-use..."
# Install with pip inside venv to avoid system package issues
pip install --upgrade pip
pip install "browser-use[cli]"

# Install Chromium
echo ""
echo "🌍 Installing Chromium browser..."
browser-use install

# Verify installation
echo ""
echo "✅ Verifying installation..."
browser-use --help > /dev/null 2>&1 && echo "✓ browser-use CLI working"

# Test browser
echo ""
echo "🧪 Testing browser launch..."
browser-use open https://example.com
sleep 2
browser-use state
browser-use close

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  browser-use open https://google.com"
echo "  browser-use state"
echo "  browser-use screenshot test.png"
echo "  browser-use close"
