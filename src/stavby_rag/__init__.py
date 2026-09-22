"""stavby-rag: local RAG over the CTU construction textbook."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stavby-rag")
except PackageNotFoundError:  # running from a plain checkout without an install
    __version__ = "0.3.0"
