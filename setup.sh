#!/bin/bash

# ReceiptIQ Setup Script for macOS
# REQUIREMENT: Python 3.12 or higher (PyTorch compatibility)
# NOTE: Python 3.13+ not yet supported (PyTorch wheels unavailable)

echo "🔧 ReceiptIQ Setup"
echo "=================="
echo ""

# Check Python version
PYTHON_VERSION=$(/usr/bin/python3.12 --version 2>/dev/null | awk '{print $2}' | cut -d. -f1-2)
if [ -z "$PYTHON_VERSION" ]; then
    echo "❌ ERROR: Python 3.12 not found!"
    echo "Please install Python 3.12 first. You can use:"
    echo "  brew install python@3.12"
    echo "  or visit: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Using Python 3.12 ($(/usr/bin/python3.12 --version))"
echo ""

# Step 1: Create virtual environment with Python 3.12
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with Python 3.12..."
    /usr/bin/python3.12 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Step 2: Activate venv
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Step 3: Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Step 3: Install dependencies from requirements.txt
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Note: transformers will handle PyTorch if needed
echo "✅ Dependencies installed!"

# Step 6: Initialize database
echo "🗄️  Initializing database..."
python scripts/init_db.py

# Final message
echo ""
echo "✅ Setup Complete!"
echo ""
echo "� Configuration Summary:"
echo "  • Python Version: 3.12+"
echo "  • PyTorch: 2.2.2 (CPU mode)"
echo "  • LLM Models: Phi-3.5-mini, Mistral-7B (downloads on first use)"
echo "  • Database: SQLite3"
echo ""
echo "🚀 To run the app, execute:"
echo "   bash run.sh"
echo ""
echo "Or manually:"
echo "   source .venv/bin/activate"
echo "   python -m app.main"
echo ""
echo "Then open: http://localhost:7860"
echo ""
echo "💡 First run will download LLM models (~2-3 min, one time only)"
