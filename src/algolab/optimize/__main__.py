"""Scheduled entry point: rank strategies and apply high-performance selection.

    python -m algolab.optimize.selection --apply
"""
import sys

from algolab.optimize.selection import main

if __name__ == "__main__":
    sys.exit(main())
