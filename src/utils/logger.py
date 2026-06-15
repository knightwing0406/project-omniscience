import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_production_logger(module_name: str) -> logging.Logger:
    """
    Creates a standardized, enterprise-grade logger for Project Omniscience.
    Streams formatted metrics to the console and archives detailed logs to a rolling file vault.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if modules hook into each other
    if logger.hasHandlers():
        return logger

    # The Telemetry Blueprint: Timestamp | Severity Level | Source Code File | Message
    log_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s.py] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Console Stream Handler (For live engineering terminal observation)
    console_stream = logging.StreamHandler(sys.stdout)
    console_stream.setFormatter(log_formatter)
    logger.addHandler(console_stream)

    # 2. Production Rolling File Handler (For archiving bugs without eating up server storage)
    # Automatically cuts off at 5 Megabytes and cycles through a maximum of 3 backup logs
    file_archive = RotatingFileHandler(
        'project_omniscience.log', 
        maxBytes=5 * 1024 * 1024, 
        backupCount=3
    )
    file_archive.setFormatter(log_formatter)
    logger.addHandler(file_archive)

    return logger

if __name__ == "__main__":
    # Local runtime configuration check
    test_logger = setup_production_logger("infrastructure_test")
    test_logger.info("⚡ [TELEMETRY] System health logging module verified. Standard pipeline active.")
    test_logger.warning("🚨 [TELEMETRY] Test alert triggered. Verifying rolling file write parameters.")
