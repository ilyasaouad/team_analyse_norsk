"""
Configuration settings for the patent analysis system.
Updated to work with environment variables and your setup.
"""

import logging
import os
import urllib.parse
from pathlib import Path
from typing import Final, Dict, Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _validate_env_vars() -> None:
    """Validate required environment variables are set."""
    required_vars = ["db_host", "db_database", "db_username", "db_password"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.warning(f"Missing environment variables: {missing}. Using defaults.")


def _build_database_url() -> str:
    """Build SQLAlchemy database URL from environment variables."""
    server = os.getenv("db_host", "localhost")
    database = os.getenv("db_database", "patent_db")
    username = os.getenv("db_username", "user")
    password = os.getenv("db_password", "password")
    driver = os.getenv("db_driver", "ODBC Driver 17 for SQL Server")

    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password}"
    )

    encoded_conn_str = urllib.parse.quote_plus(conn_str)
    return f"mssql+pyodbc:///?odbc_connect={encoded_conn_str}"


class Config:
    """Central configuration class for the patent analysis system.

    Environment Variables:
        db_host: Database server hostname (default: localhost)
        db_database: Database name (default: patent_db)
        db_username: Database username (default: user)
        db_password: Database password (default: password)
        db_driver: ODBC driver name (default: ODBC Driver 17 for SQL Server)
        LOG_LEVEL: Logging level (default: INFO)
    """

    # Database
    DATABASE_CONFIG: Final[Dict[str, Any]] = {
        "host": os.getenv("db_host", "localhost"),
        "database": os.getenv("db_database", "patent_db"),
        "username": os.getenv("db_username", "user"),
        "password": os.getenv("db_password", "password"),
        "driver": os.getenv("db_driver", "ODBC Driver 17 for SQL Server"),
    }
    DB_URL: Final[str] = _build_database_url()
    DB_ECHO: Final[bool] = os.getenv("DB_ECHO", "False").lower() == "true"
    DB_POOL_SIZE: Final[int] = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: Final[int] = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    # Processing
    BATCH_SIZE: Final[int] = 200
    OUTPUT_DIR: Final[Path] = Path(os.getenv("OUTPUT_DIR", "output"))

    # API configuration
    API_RATE_LIMIT: Final[int] = 1000
    TIMEOUT_SECONDS: Final[int] = 30

    # Analysis thresholds
    MIN_FAMILY_SIZE: Final[int] = 1
    MAX_FAMILY_SIZE: Final[int] = 1000

    # File naming conventions
    CSV_FILENAME_FORMAT: Final[str] = (
        "{country_code}_{start_year}_{end_year}_analysis.csv"
    )
    LOG_FILENAME: Final[str] = "patent_analysis.log"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def initialize(cls) -> None:
        """Initialize configuration and create required directories.

        This should be called once at application startup.
        Sets up logging and ensures output directories exist.
        """
        _validate_env_vars()

        # Create output directory
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format=cls.LOG_FORMAT,
            handlers=[
                logging.FileHandler(cls.OUTPUT_DIR / cls.LOG_FILENAME),
                logging.StreamHandler(),
            ],
        )

        logger.info("Configuration initialized")
        logger.info(f"Output directory: {cls.OUTPUT_DIR.resolve()}")
        logger.info(f"Log level: {cls.LOG_LEVEL}")

    @classmethod
    def update(cls, **kwargs) -> None:
        """Update configuration values dynamically.

        Only allows updating specific safe parameters.

        Args:
            **kwargs: Configuration parameters to update
                (batch_size, output_dir, log_level, timeout_seconds)

        Raises:
            ValueError: If parameter is invalid or unsupported

        Example:
            >>> Config.update(batch_size=500, log_level='DEBUG')
        """
        ALLOWED_KEYS = {"batch_size", "output_dir", "log_level", "timeout_seconds"}

        for key, value in kwargs.items():
            key_lower = key.lower()
            if key_lower not in ALLOWED_KEYS:
                raise ValueError(f"'{key}' is not updatable. Allowed: {ALLOWED_KEYS}")

            # Validate and set
            if key_lower == "batch_size":
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("batch_size must be positive integer")
                cls.BATCH_SIZE = value
                logger.info(f"Updated batch_size to {value}")

            elif key_lower == "output_dir":
                output_dir = Path(value)
                output_dir.mkdir(parents=True, exist_ok=True)
                cls.OUTPUT_DIR = output_dir
                logger.info(f"Updated output_dir to {output_dir}")

            elif key_lower == "log_level":
                valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
                if value not in valid_levels:
                    raise ValueError(f"log_level must be one of {valid_levels}")
                cls.LOG_LEVEL = value
                logger.setLevel(value)
                logger.info(f"Updated log_level to {value}")

            elif key_lower == "timeout_seconds":
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("timeout_seconds must be positive integer")
                cls.TIMEOUT_SECONDS = value
                logger.info(f"Updated timeout_seconds to {value}")

    @classmethod
    def get_database_url(cls, safe: bool = False) -> str:
        """Get database connection URL.

        Args:
            safe: If True, return URL without password (for logging/debugging)

        Returns:
            SQLAlchemy database URL string

        Example:
            >>> url = Config.get_database_url()  # With password
            >>> safe_url = Config.get_database_url(safe=True)  # Without password
        """
        if safe:
            return f"mssql+pyodbc://{cls.DATABASE_CONFIG['host']}/{cls.DATABASE_CONFIG['database']}"

        return cls.DB_URL
