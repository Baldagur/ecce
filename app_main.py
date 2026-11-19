"""Entry point for the packaged app.
No arguments (or a .psd path) -> PSD Prep window (double-click).
Other arguments               -> command line, e.g.  PSDPrep in.psd --out-flat out.png
"""
import sys

from psdprep.cli import main

if __name__ == "__main__":
    sys.exit(main())
