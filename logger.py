import logging
import logging.config

# Define the logger configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,  # Disable existing loggers
    "formatters": {  # Define log message formats
        "detailed": {  # Detailed format for file logs
            "format": "%(levelname)s - %(message)s"
        },
        "simple": {  # Simple format for console logs
            "format": "%(levelname)s - %(message)s"
        },
        "colored": {  # Colorized format for console logs
            "()": "colorlog.ColoredFormatter",
            "format": "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        },
    },
    "handlers": {  # Define log handlers
        "console": {  # Console handler
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "colored",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {  # Root logger configuration
        "level": "INFO",
        "handlers": ["console"],  # Use both console and file handlers
    },
}

# Apply the logger configuration
logging.config.dictConfig(LOG_CONFIG)

# Create a logger instance
alfred_logger = logging.getLogger(__name__)
