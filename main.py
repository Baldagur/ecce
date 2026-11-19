"""Entry point for the double-click app (and PyInstaller)."""
import sys

from psdprep.cli import main

if __name__ == "__main__":
    sys.exit(main())
