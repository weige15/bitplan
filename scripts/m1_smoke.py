#!/usr/bin/env python3
"""Repository entry point for the dependency-free M1 smoke test."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitplan.smoke import main

raise SystemExit(main())
