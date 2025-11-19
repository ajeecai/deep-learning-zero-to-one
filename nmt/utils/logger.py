import logging
import colorlog


def setup_logger(name="NMT", log_file=None, level=logging.INFO):
    """
    Set up a logger with console (colored) and file handlers.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If None, no file logging.
        level (int, optional): The minimum logging level to display on the console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Logger is already configured, don't add more handlers
        return logger

    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # --- Console Handler with Colors ---
    ch = colorlog.StreamHandler()
    ch.setLevel(level)
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        reset=True,
        style="%",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --- File Handler (optional) ---
    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # Log everything to the file
        fh_formatter = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
