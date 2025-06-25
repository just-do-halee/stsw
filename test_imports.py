#!/usr/bin/env python3
"""Test if imports work correctly."""

import sys
print(f"Python version: {sys.version}")

try:
    import stsw
    print("✓ stsw imported successfully")
    print(f"  Version: {stsw.__version__}")
except Exception as e:
    print(f"✗ Failed to import stsw: {e}")
    sys.exit(1)

try:
    from stsw import StreamWriter, StreamReader, TensorMeta
    print("✓ Main classes imported successfully")
except Exception as e:
    print(f"✗ Failed to import main classes: {e}")
    sys.exit(1)

print("\nAll imports successful!")