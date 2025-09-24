#!/bin/bash
# Setup script for installing pre-commit hooks

set -e  # Exit on error

echo "Setting up pre-commit hooks for tt_lazy project..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not found in PATH"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Install development dependencies
echo "Installing development dependencies..."
pip3 install -r tools/requirements-dev.txt

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install --config tools/.pre-commit-config.yaml

# Run pre-commit on all files to check current state (optional)
echo "Running pre-commit checks on all files..."
echo "Note: This may take a moment and might show formatting changes needed."
pre-commit run --all-files --config tools/.pre-commit-config.yaml || true

echo ""
echo "✅ Pre-commit hooks have been installed successfully!"
echo ""
echo "The following hooks are now active:"
echo "  - clang-format (C++ formatting)"
echo "  - trailing-whitespace removal"
echo "  - end-of-file fixer"
echo "  - YAML checker"
echo "  - Large file checker"
echo "  - Mixed line ending fixer"
echo "  - Merge conflict checker"
echo "  - Black (Python formatting)"
echo "  - isort (Python import sorting)"
echo "  - flake8 (Python linting)"
echo ""
echo "These hooks will run automatically on 'git commit'."
echo "To run manually on all files: pre-commit run --all-files"
echo "To run on specific files: pre-commit run --files <file1> <file2> ..."
