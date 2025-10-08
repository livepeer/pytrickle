#!/usr/bin/env python3
"""
Lint runner script for pytrickle project.
Runs all configured linting tools with proper 4-space indentation and POSIX line endings.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔍 Running {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} passed")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Run all linting tools."""
    project_root = Path(__file__).parent
    python_files = ["pytrickle", "tests", "examples"]
    
    # Change to project root
    import os
    os.chdir(project_root)
    
    success = True
    
    # Run Black formatter
    success &= run_command(
        ["black", "--check", "--diff"] + python_files,
        "Black code formatting check"
    )
    
    # Run isort import sorting
    success &= run_command(
        ["isort", "--check-only", "--diff"] + python_files,
        "isort import sorting check"
    )
    
    # Run Ruff linter
    success &= run_command(
        ["ruff", "check"] + python_files,
        "Ruff linting"
    )
    
    # Run Ruff formatter
    success &= run_command(
        ["ruff", "format", "--check"] + python_files,
        "Ruff formatting check"
    )
    
    # Run MyPy type checking
    success &= run_command(
        ["mypy"] + python_files,
        "MyPy type checking"
    )
    
    if success:
        print("\n🎉 All linting checks passed!")
        return 0
    else:
        print("\n💥 Some linting checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
