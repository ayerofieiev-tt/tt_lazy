#!/usr/bin/env python3
"""
Test runner for TT Lazy Python bindings
"""

import os
import sys

from test_python_bindings import main

# Add the project root to Python path to import the module
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

if __name__ == "__main__":
    sys.exit(main())
