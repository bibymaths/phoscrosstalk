"""
logger.py
A unified, rich-text logger for the PhosCrosstalk pipeline.
Handles colorful console output and structured file logging.
"""
import logging
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Define custom-theme for consistent coloring
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "success": "bold green",
    "header": "bold magenta"
})

console = Console(theme=custom_theme)

class RichLogger:
    _instance = None

    def __new__(cls, name="PhosCrosstalk", log_file="pipeline.log", level=logging.INFO):
        if cls._instance is None:
            cls._instance = super(RichLogger, cls).__new__(cls)
            cls._instance._setup(name, log_file, level)
        return cls._instance

    def _setup(self, name, log_file, level):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers

        # 1. Console Handler (Rich)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True
        )
        rich_handler.setLevel(level)
        self.logger.addHandler(rich_handler)

        # 2. File Handler (Standard Text)
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, mode="w")
            file_handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def success(self, msg, *args, **kwargs):
        """Custom success level (mapped to INFO with styling)"""
        console.print(f"[success]✔ {msg}[/success]")
        # Log to file as INFO
        self.logger.info(f"[SUCCESS] {msg}", *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def header(self, msg):
        """Prints a styled header block."""
        console.print()
        console.rule(f"[header]{msg}[/header]")
        self.logger.info(f"=== {msg} ===")

    def get_console(self):
        """Returns the raw Rich console for advanced use (tables, progress)."""
        return console

# Global singleton accessor
def get_logger(log_file=None):
    return RichLogger(log_file=log_file)