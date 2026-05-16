#!/usr/bin/env python3
"""CLI script to test Donut extraction on a single receipt image.

Usage:
    python scripts/test_donut_extraction.py <image_path> [--task sroie|cord]

Example:
    python scripts/test_donut_extraction.py data/cord_100/images/receipt_001.png --task sroie
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tools.donut import extract_fields_donut


def main():
    parser = argparse.ArgumentParser(description="Test Donut extraction on receipt image")
    parser.add_argument("image_path", help="Path to receipt image")
    parser.add_argument("--task", choices=["sroie", "cord"], default="sroie", help="Task type")
    parser.add_argument("--verbose", action="store_true", help="Print raw output")
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    print(f"🔍 Testing Donut extraction on: {image_path}")
    print(f"   Task: {args.task}")
    print()
    
    result = extract_fields_donut(image_path=str(image_path), task=args.task)
    
    print("📋 Extracted Fields:")
    print(f"   Vendor: {result.get('vendor')}")
    print(f"   Date: {result.get('date')}")
    print(f"   Total: {result.get('total')}")
    print(f"   Subtotal: {result.get('subtotal')}")
    print(f"   Tax: {result.get('tax')}")
    print()
    print(f"📊 Metadata:")
    print(f"   Source: {result.get('extraction_source')}")
    print(f"   Confidence: {result.get('confidence', 0):.2f}")
    print()
    
    if args.verbose:
        print("📄 Raw Output:")
        print("-" * 60)
        print(result.get('raw', 'N/A'))
        print("-" * 60)
    
    # Success if any field extracted
    has_data = any(result.get(f) for f in ['vendor', 'date', 'total'])
    sys.exit(0 if has_data else 1)


if __name__ == "__main__":
    main()
