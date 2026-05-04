import datetime
import re
import sys
import time
from urllib.parse import quote

import pyfiglet
from rich.console import Console

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
OSC8_RE = re.compile(r"\x1b]8;;.*?\x1b\\(.*?)\x1b]8;;\x1b\\", re.DOTALL)


def visible_len(text: str) -> int:
    """Return printable width after removing ANSI color and OSC8 hyperlink escapes."""
    text = OSC8_RE.sub(r"\1", text)
    text = ANSI_RE.sub("", text)
    return len(text)


def printable_clip(text: str, max_width: int) -> str:
    """
    Clip text by visible width while preserving escape sequences.
    Assumes escapes are well-formed ANSI SGR or OSC8 hyperlinks.
    """
    result = []
    visible = 0
    i = 0
    n = len(text)

    while i < n and visible < max_width:
        # ANSI SGR
        if text[i] == "\x1b" and i + 1 < n and text[i + 1] == "[":
            m = ANSI_RE.match(text, i)
            if m:
                result.append(m.group(0))
                i = m.end()
                continue

        # OSC8 hyperlink
        if text[i] == "\x1b" and text[i : i + 5] == "\x1b]8;;":
            end_meta = text.find("\x1b\\", i)
            if end_meta == -1:
                break
            start_label = end_meta + 2
            close_seq = "\x1b]8;;\x1b\\"
            end_label = text.find(close_seq, start_label)
            if end_label == -1:
                break

            prefix = text[i:start_label]
            label = text[start_label:end_label]
            suffix = close_seq

            remaining = max_width - visible
            clipped_label = label[:remaining]
            result.append(prefix + clipped_label + suffix)
            visible += len(clipped_label)
            i = end_label + len(close_seq)
            continue

        result.append(text[i])
        visible += 1
        i += 1

    return "".join(result)


def print_logo(
    name: str,
    version: str = "",
    tagline: str = "",
    author: str = "",
    email: str = "",
    orcid: str = "",
    website: str = "",
    font: str = "slant",
    color: str = "bright_green",
    animate: bool = True,
):
    """
    Animated terminal logo with clickable email, ORCID, and website links.
    """
    console = Console(width=86)
    W = 80
    INNER = W - 2

    art_raw = pyfiglet.figlet_format(name, font=font).rstrip("\n")
    art_lines = art_raw.splitlines() if art_raw else [name]

    def osc8(url: str, label: str) -> str:
        return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"

    def ansi_code(c: str) -> str:
        return {
            "bright_green": "92",
            "cyan": "96",
            "magenta": "95",
            "green": "32",
            "blue": "34",
            "white": "37",
            "yellow": "93",
            "red": "91",
        }.get(c, "92")

    def normalize_website(url: str) -> str:
        if not url:
            return ""
        if url.startswith(("http://", "https://")):
            return url
        return f"https://{url}"

    def normalize_orcid(value: str) -> str:
        if not value:
            return ""
        if value.startswith(("http://", "https://")):
            return value
        return f"https://orcid.org/{value}"

    def normalize_email(value: str) -> str:
        if not value:
            return ""
        return f"mailto:{quote(value)}"

    cc = ansi_code(color)

    def sleep(delay: float) -> None:
        if animate:
            time.sleep(delay)

    def framed_write(content: str, delay: float = 0.0) -> None:
        clipped = printable_clip(content, INNER)
        pad = " " * max(0, INNER - visible_len(clipped))
        sys.stdout.write(f"\033[{cc}m┃\033[0m{clipped}{pad}\033[{cc}m┃\033[0m\n")
        sys.stdout.flush()
        sleep(delay)

    def centered_write(content: str = "", delay: float = 0.0) -> None:
        clipped = printable_clip(content, INNER)
        pad_total = max(0, INNER - visible_len(clipped))
        left = pad_total // 2
        right = pad_total - left
        sys.stdout.write(
            f"\033[{cc}m┃\033[0m{' ' * left}{clipped}{' ' * right}\033[{cc}m┃\033[0m\n"
        )
        sys.stdout.flush()
        sleep(delay)

    def hline(left: str = "┏", right: str = "┓", delay: float = 0.002) -> None:
        sys.stdout.write(f"\033[{cc}m{left}\033[0m")
        for ch in "━" * W:
            sys.stdout.write(f"\033[{cc}m{ch}\033[0m")
            sys.stdout.flush()
            sleep(delay)
        sys.stdout.write(f"\033[{cc}m{right}\033[0m\n")
        sys.stdout.flush()

    def separator(delay: float = 0.001) -> None:
        sys.stdout.write(f"\033[{cc}m┣{'━' * W}┫\033[0m\n")
        sys.stdout.flush()
        sleep(delay)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info_lines = []
    if version:
        info_lines.append(f" Version : {version}")
    if tagline:
        info_lines.append(f" Tagline : {tagline}")
    if author:
        info_lines.append(f" Author  : {author}")
    if email:
        info_lines.append(f" Email   : {osc8(normalize_email(email), email)}")
    if orcid:
        orcid_url = normalize_orcid(orcid)
        orcid_label = orcid.replace("https://orcid.org/", "").replace(
            "http://orcid.org/", ""
        )
        info_lines.append(f" ORCID   : {osc8(orcid_url, orcid_label)}")
    if website:
        web_url = normalize_website(website)
        web_label = website.replace("https://", "").replace("http://", "")
        info_lines.append(f" Website : {osc8(web_url, web_label)}")
    info_lines.append(f" Date    : {timestamp}")

    hline()
    centered_write()
    for line in art_lines:
        centered_write(line, delay=0.008 if animate else 0.0)
    centered_write()
    separator()
    for line in info_lines:
        framed_write(line, delay=0.006 if animate else 0.0)
    hline("┗", "┛")

    # Reset any lingering formatting explicitly
    sys.stdout.write("\033[0m")
    sys.stdout.flush()
