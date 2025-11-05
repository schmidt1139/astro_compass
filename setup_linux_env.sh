#!/bin/bash
# Setup script for astro_compass on a shared Linux system
# This script ensures proper virtual environment isolation

set -e  # Exit on error

echo "=========================================="
echo "astro_compass Setup Script"
echo "For Shared Linux Systems"
echo "=========================================="
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "📁 Project directory: $PROJECT_DIR"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found: Python $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "   ❌ ERROR: Python 3.8 or higher required"
    exit 1
fi
echo "   ✅ Python version OK"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "   ✅ Virtual environment created at: $PROJECT_DIR/venv"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Verify we're in the venv
PYTHON_PATH=$(which python)
if [[ "$PYTHON_PATH" == *"$PROJECT_DIR/venv"* ]]; then
    echo "   ✅ Using isolated Python: $PYTHON_PATH"
else
    echo "   ⚠️  WARNING: Not using venv Python!"
    echo "   Current: $PYTHON_PATH"
    echo "   Expected: $PROJECT_DIR/venv/bin/python"
fi
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
echo "   ✅ pip upgraded to version: $(pip --version | awk '{print $2}')"
echo ""

# Install project dependencies
echo "📚 Installing project dependencies..."
echo "   This will install: gymnasium, stable-baselines3, torch, matplotlib, numpy"
echo "   Location: $PROJECT_DIR/venv/lib/python*/site-packages/"
echo ""

pip install -e . 

echo ""
echo "✅ Installation complete!"
echo ""

# Verify key imports
echo "🧪 Testing key imports..."
python -c "import gymnasium; print('   ✅ gymnasium:', gymnasium.__version__)" 2>/dev/null || echo "   ❌ gymnasium failed"
python -c "import torch; print('   ✅ torch:', torch.__version__)" 2>/dev/null || echo "   ❌ torch failed"
python -c "import matplotlib; print('   ✅ matplotlib:', matplotlib.__version__)" 2>/dev/null || echo "   ❌ matplotlib failed"
python -c "import numpy; print('   ✅ numpy:', numpy.__version__)" 2>/dev/null || echo "   ❌ numpy failed"
python -c "from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env; print('   ✅ TwoBodyRendezvous_Env imported successfully')" 2>/dev/null || echo "   ⚠️  TwoBodyRendezvous_Env import failed"
echo ""

# Show isolation info
echo "=========================================="
echo "✅ Setup Complete - Your Environment is Isolated!"
echo "=========================================="
echo ""
echo "📊 Environment Summary:"
echo "   User: $(whoami)"
echo "   Python: $PYTHON_PATH"
echo "   Packages: $(pip list | wc -l) installed"
echo "   Location: $PROJECT_DIR/venv/"
echo ""
echo "🔒 Isolation Status:"
echo "   ✅ Your packages are in: $PROJECT_DIR/venv/lib/"
echo "   ✅ System Python unaffected"
echo "   ✅ Other users unaffected"
echo ""
echo "📝 Next Steps:"
echo "   1. Always activate before working:"
echo "      $ source venv/bin/activate"
echo ""
echo "   2. Run your scripts:"
echo "      $ python src/scripts/datagen_Hamiltonian_TBR_controller.py"
echo ""
echo "   3. Run tests:"
echo "      $ python -m pytest tests/ -v"
echo ""
echo "   4. Deactivate when done:"
echo "      $ deactivate"
echo ""
echo "💡 Tip: Add this to ~/.bashrc for quick access:"
echo "   alias activate_astro='cd $PROJECT_DIR && source venv/bin/activate'"
echo ""

# Create a simple activation helper
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Quick activation script
cd "$(dirname "${BASH_SOURCE[0]}")"
source venv/bin/activate
echo "✓ astro_compass environment activated"
echo "  Python: $(which python)"
echo "  To deactivate: type 'deactivate'"
EOF
chmod +x activate_env.sh

echo "✅ Created helper script: ./activate_env.sh"
echo "   Use: source ./activate_env.sh (instead of source venv/bin/activate)"
echo ""
