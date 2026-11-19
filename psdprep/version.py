"""Build stamp shown in the window title and the self-test.
build.py writes _build_stamp.py inside packaged apps; from source it's 'dev'."""
try:
    from ._build_stamp import BUILD  # type: ignore  # written by build.py
except ImportError:
    BUILD = "dev"
