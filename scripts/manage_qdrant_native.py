#!/usr/bin/env python
"""CLI wrapper for the Windows native Qdrant lifecycle manager."""

from src.production.native_qdrant import main

if __name__ == "__main__":
    raise SystemExit(main())
