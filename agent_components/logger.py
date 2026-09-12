import logging
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "sanctions_screening.log"

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | [sanctions] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger() -> logging.Logger:
    log = logging.getLogger("sanctions_screening")
    log.setLevel(logging.INFO)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    # File handler — dedup by resolved path
    has_file = any(
        isinstance(h, logging.FileHandler)
        and Path(h.baseFilename).resolve() == LOG_PATH.resolve()
        for h in log.handlers
    )
    if not has_file:
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(formatter)
        log.addHandler(fh)

    # Console handler — dedup by stream
    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, "stream", None) is sys.stdout
        for h in log.handlers
    )
    if not has_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        log.addHandler(sh)

    log.propagate = False  # don't bubble to root (the deep research pipeline owns root)
    return log
