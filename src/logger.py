import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    logger = logging.getLogger("app")

    if logger.hasHandlers():
        return logger

    level = os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # file
    fh = RotatingFileHandler("app.log", maxBytes=1000000, backupCount=3)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger