import logging
import os

def setup_logger():
    level = os.getenv("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    return logging.getLogger("app")

import uuid

def log_error(message, context=None):
    error_id = str(uuid.uuid4())
    logging.error(f"[{error_id}] {message} | context={context}")
    return error_id