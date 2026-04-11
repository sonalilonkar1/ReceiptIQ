#!/bin/bash

# ReceiptIQ Setup Script for macOS

echo "🔧 ReceiptIQ Setup"
echo "=================="

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

echo "✅ Activating virtual environment..."
source .venv/bin/activate

echo "� Installing PyTorch (CPU) from official index..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet 2>/dev/null || {
    echo "⚠️  PyTorch CPU installation failed, trying with output..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

echo "📚 Installing other dependencies..."
pip install --upgrade pip --quiet
pip install gradio pydantic python-dotenv requests beautifulsoup4 pytesseract Pillow pypdf transformers accelerate bitsandbytes sentencepiece --quiet 2>/dev/null || {
    echo "Some packages failed to install, continuing with available packages..."
}

echo "🗄️  Initializing database..."
python scripts/init_db.py 2>/dev/null || echo "Database initialization may need attention"

echo ""
echo "✅ Setup Attempt Complete!"
echo ""
echo "🚀 To run the app:"
echo "   source .venv/bin/activate"
echo "   python -m app.main"
echo ""
echo "Then open: http://localhost:7860"
