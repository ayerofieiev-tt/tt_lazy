#!/usr/bin/env python3
"""
Test runner for TT Lazy Python bindings
"""

import sys
import os

# Add the project root to Python path to import the module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import and run the main test
from test_python_bindings import main

if __name__ == "__main__":
    sys.exit(main())
