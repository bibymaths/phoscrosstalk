"""
logger.py
A unified, rich-text logger for the PhosCrosstalk pipeline.
Handles colorful console output and structured file logging.
"""

import datetime
import logging
import os
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Define custom-theme for consistent coloring
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "success": "bold green",
        "header": "bold magenta",
    }
)

console = Console(theme=custom_theme)


class RichLogger:
    """
    A Singleton wrapper around Python's logging module that provides rich-text
    console output and structured file logging.

    Ensures that only one logger instance exists throughout the application lifecycle,
    managing output styles (colors, emojis) via the `rich` library while maintaining
    standard text logs in a file.
    """

    _instance = None

    def __new__(cls, name="PhosCrosstalk", log_file="pipeline.log", level=logging.INFO):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup(name, log_file, level)
        return cls._instance

    def _setup(self, name, log_file, level):
        """
        Initializes the logger configuration, handlers, and formatters.

        Sets up two handlers:
        1. A `RichHandler` for colorful, high-readability console output.
        2. A `FileHandler` for persistent, standard-formatted logging.

        Args:
            name (str): Name of the logger instance.
            log_file (str): Path to the output log file.
            level (int): Logging threshold (e.g., logging.INFO).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers

        # 1. Console Handler (Rich)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        rich_handler.setLevel(level)
        self.logger.addHandler(rich_handler)

        # 2. File Handler (Handle None case explicitly)
        if log_file:
            self.add_file_handler(log_file, level)

    def add_file_handler(self, log_file, level=logging.INFO):
        """
        Safely adds a file handler to an existing logger.
        Avoids duplicate file handlers if called multiple times.
        """
        # Remove existing FileHandlers to avoid duplicate logs if re-configured
        self.logger.handlers = [
            h for h in self.logger.handlers if not isinstance(h, logging.FileHandler)
        ]

        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(path, mode="w")
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, msg, *args, **kwargs):
        """
        Log an informational message.

        Args:
            msg (str): The message string.
            flush (bool, optional): If True, flush all handlers after logging.
            *args, **kwargs: Arguments passed to the standard logger.
        """
        flush = kwargs.pop("flush", False)  # accept & remove flush kwarg
        self.logger.info(msg, *args, **kwargs)
        if flush:
            # flush all handlers (file handler, rich handler, etc.)
            for handler in self.logger.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

    def success(self, msg, *args, **kwargs):
        """
        Log a success message with a checkmark icon.

        Args:
            msg (str): The message string.
            *args, **kwargs: Arguments passed to the standard logger.
        """
        console.print(f"[success]✔ {msg}[/success]")
        # Log to file as INFO
        self.logger.info(f"[SUCCESS] {msg}", *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Log a warning message (yellow styling).

        Args:
            msg (str): The warning message.
            *args, **kwargs: Arguments passed to the standard logger.
        """
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Log an error message (red styling).

        Args:
            msg (str): The error message.
            *args, **kwargs: Arguments passed to the standard logger.
        """
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Log a critical error message (bold red/background styling).

        Args:
            msg (str): The critical message.
            *args, **kwargs: Arguments passed to the standard logger.
        """
        self.logger.critical(msg, *args, **kwargs)

    def header(self, msg):
        """
        Print a styled horizontal rule with a centered header title.

        Useful for visually separating distinct stages of the pipeline in the console output.
        Also logs the header text to the file as an INFO message.

        Args:
            msg (str): The header title text.
        """  # noqa: E501
        console.print()
        console.rule(f"[header]{msg}[/header]")
        self.logger.info(f"=== {msg} ===")

    def get_console(self):
        """
        Retrieve the underlying `rich.console.Console` instance.

        Allows access to advanced `rich` features like progress bars, tables,
        and live displays that are not covered by standard logging methods.

        Returns:
            rich.console.Console: The active console object.
        """
        return console


# Global singleton accessor
def get_logger(log_file="pipeline.log", timestamp=True):
    """
    Return the process-wide ``RichLogger`` singleton.

    This function is **idempotent**: once the singleton has been created by
    the first call (typically from ``main.py`` or the first imported module),
    all subsequent calls return the same instance regardless of the arguments
    passed.  This prevents:

    * Duplicate ``FileHandler`` / ``StreamHandler`` additions when multiple
      modules call ``get_logger()`` at import time.
    * Multiple timestamped ``.log`` files being created within a single
      pipeline run or Streamlit session (because every call would otherwise
      generate a fresh timestamp and attempt to set up a new handler).

    Singleton / handler policy
    --------------------------
    * The first call wins: it creates ``RichLogger`` with the supplied
      ``log_file`` name (optionally timestamped) and attaches both a
      ``RichHandler`` (console) and a ``FileHandler``.
    * All subsequent calls return the existing instance untouched.
    * If you need a *different* log file for a new session (e.g., a second
      CLI invocation in the same interpreter), reset ``RichLogger._instance``
      to ``None`` before calling ``get_logger()`` again.  This is intentional
      and explicit rather than automatic, to avoid accidental log-file
      proliferation.

    Streamlit note
    --------------
    Streamlit reruns do **not** reimport already-cached modules, so the
    singleton persists across reruns within the same server process.  A new
    timestamped log file is therefore created only when the server process
    starts, not on every page rerender.

    Args:
        log_file (str): Base filename used only on the *first* call.
        timestamp (bool): If ``True`` (default), a ``YYYY-MM-DD_HH-MM-SS``
            suffix is appended to the base filename on the first call.
            Subsequent calls ignore this argument entirely.

    Returns:
        RichLogger: The singleton logger instance.
    """
    # Fast path: singleton already initialised – return immediately without
    # generating a new timestamp or adding any handlers.
    if RichLogger._instance is not None:
        return RichLogger._instance

    if log_file is None:
        log_file = "pipeline.log"

    if timestamp:
        # Generate a timestamped filename only once, at singleton creation.
        # Format: pipeline_2023-10-27_15-30-00.log
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root, ext = os.path.splitext(log_file)
        if not ext:
            ext = ".log"
        log_file = f"{root}_{ts}{ext}"

    return RichLogger(log_file=log_file)
