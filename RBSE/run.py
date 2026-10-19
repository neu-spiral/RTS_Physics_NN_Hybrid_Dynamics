#!/usr/bin/env python
# run.py - CLI entry point for RBSE training.
#
# Usage:
#   python RBSE/run.py --models ho
#   python RBSE/run.py --models pendulum --epochs 3
#   python RBSE/run.py --models ho retina
#   python RBSE/run.py --models all
#
# Or as a module:
#   python -m RBSE --models ho

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RBSE.train import main

if __name__ == "__main__":
    main()
