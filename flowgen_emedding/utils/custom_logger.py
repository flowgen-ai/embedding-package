# utils/logging_config.py
import os
import logging


def setup_logging(service_name: str, log_dir: str = 'logs', log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure and setup logging for a service.

    Args:
        service_name (str): Name of the service (used for both logger name and log file name)
        log_dir (str): Directory where log files will be stored
        log_level (int): Logging level (default: logging.INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Configure log file path
    log_file = os.path.join(log_dir, f'{service_name}.log')

    # Configure logging format
    log_format = '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'

    # Configure handlers
    handlers = [
        logging.StreamHandler(),
      # logging.FileHandler(log_file)
    ]

    # Apply configuration
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

    return logging.getLogger(service_name)