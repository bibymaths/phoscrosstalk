"""
CLI entry point for launching the PhosCrosstalk Streamlit app.
"""

from pathlib import Path
import sys

from streamlit.web import cli as stcli


def cli() -> None:
    app_path = Path(__file__).with_name("app.py")

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
    ]

    stcli.main()
