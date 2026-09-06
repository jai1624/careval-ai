"""
Structured logging setup. Uses Google Cloud Logging in GCP mode when
available, and falls back to standard stdout logging everywhere else
(including if Cloud Logging client setup itself fails -- logging must
never be the thing that crashes the app).
"""
from __future__ import annotations

import logging
import sys

from src.config import config

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger("careval")
    root.setLevel(config.log_level)

    handler_attached = False

    if config.is_gcp:
        try:
            import google.cloud.logging as gcloud_logging

            client = gcloud_logging.Client()
            client.setup_logging(log_level=getattr(logging, config.log_level, logging.INFO))
            handler_attached = True
        except Exception:
            logging.getLogger("careval.logging_setup").warning(
                "Cloud Logging unavailable, falling back to stdout logging", exc_info=True
            )

    if not handler_attached:
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    _configured = True
