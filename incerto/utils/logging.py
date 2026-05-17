"""
Logging utilities for incerto.

Provides a centralized logging configuration for the library.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> from incerto.utils.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Calibrating model...")
    """
    return logging.getLogger(f"incerto.{name}")


def setup_logging(
    level: int = logging.INFO,
    format_str: str | None = None,
    handlers: list | None = None,
) -> None:
    """
    Configure logging for incerto.

    Args:
        level: Logging level (default: INFO)
        format_str: Custom format string (default: includes timestamp, level, name, message)
        handlers: List of handlers (default: StreamHandler to stdout)

    Example:
        >>> from incerto.utils.logging import setup_logging
        >>> import logging
        >>> setup_logging(level=logging.DEBUG)
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if handlers is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_str))
        handlers = [handler]

    # Configure root incerto logger
    logger = logging.getLogger("incerto")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    for handler in handlers:
        logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False


def disable_logging() -> None:
    """
    Disable all logging from incerto.

    Example:
        >>> from incerto.utils.logging import disable_logging
        >>> disable_logging()
    """
    logging.getLogger("incerto").setLevel(logging.CRITICAL + 1)


# Default configuration
setup_logging(level=logging.WARNING)
