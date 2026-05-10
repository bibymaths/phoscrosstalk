import datetime
import re
import shutil
import sys
import time
from urllib.parse import quote

import pyfiglet

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
OSC8_RE = re.compile(r"\x1b]8;;.*?\x1b\\(.*?)\x1b]8;;\x1b\\", re.DOTALL)


def visible_len(text: str) -> int:
    """Return printable width after removing ANSI color and OSC8 hyperlink escapes."""
    text = OSC8_RE.sub(r"\1", text)
    text = ANSI_RE.sub("", text)
    return len(text)


def printable_clip(text: str, max_width: int) -> str:
    """
    Clip text by visible width while preserving ANSI SGR and OSC8 hyperlinks.
    """
    result = []
    visible = 0
    i = 0
    n = len(text)

    while i < n and visible < max_width:
        # ANSI SGR escape
        if text[i] == "\x1b" and i + 1 < n and text[i + 1] == "[":
            m = ANSI_RE.match(text, i)
            if m:
                result.append(m.group(0))
                i = m.end()
                continue

        # OSC8 hyperlink escape
        if text[i] == "\x1b" and text[i: i + 5] == "\x1b]8;;":
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
        min_width: int = 80,
        max_width: int = 120,
):
    """
    Animated terminal logo with centered figlet art and safe borders.

    The frame width is automatically chosen from:
    - terminal width
    - figlet logo width
    - metadata line width
    """

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

    def sleep(delay: float) -> None:
        if animate and delay > 0:
            time.sleep(delay)

    cc = ansi_code(color)

    art_raw = pyfiglet.figlet_format(name, font=font).rstrip("\n")
    art_lines = art_raw.splitlines() if art_raw else [name]

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

    terminal_width = shutil.get_terminal_size((100, 24)).columns

    widest_art = max(visible_len(line) for line in art_lines)
    widest_info = max((visible_len(line) for line in info_lines), default=0)

    content_width = max(widest_art, widest_info)
    desired_width = content_width + 6  # padding + borders

    frame_width = max(min_width, desired_width)
    frame_width = min(frame_width, max_width, terminal_width)

    inner_width = frame_width - 2

    # If terminal is very narrow, keep frame mathematically valid.
    if inner_width < 10:
        inner_width = 10
        frame_width = inner_width + 2

    def colorize(text: str) -> str:
        return f"\033[{cc}m{text}\033[0m"

    def hline(left: str = "┏", right: str = "┓", delay: float = 0.001) -> None:
        sys.stdout.write(colorize(left))

        for ch in "━" * inner_width:
            sys.stdout.write(colorize(ch))
            sys.stdout.flush()
            sleep(delay)

        sys.stdout.write(colorize(right) + "\n")
        sys.stdout.flush()

    def separator(delay: float = 0.001) -> None:
        sys.stdout.write(colorize("┣" + "━" * inner_width + "┫") + "\n")
        sys.stdout.flush()
        sleep(delay)

    def framed_write(content: str = "", delay: float = 0.0) -> None:
        clipped = printable_clip(content, inner_width)
        pad = inner_width - visible_len(clipped)

        sys.stdout.write(
            colorize("┃")
            + clipped
            + (" " * max(0, pad))
            + colorize("┃")
            + "\n"
        )
        sys.stdout.flush()
        sleep(delay)

    def centered_write(content: str = "", delay: float = 0.0) -> None:
        clipped = printable_clip(content, inner_width)
        text_width = visible_len(clipped)

        pad_total = inner_width - text_width
        left_pad = max(0, pad_total // 2)
        right_pad = max(0, pad_total - left_pad)

        sys.stdout.write(
            colorize("┃")
            + (" " * left_pad)
            + clipped
            + (" " * right_pad)
            + colorize("┃")
            + "\n"
        )
        sys.stdout.flush()
        sleep(delay)

    hline()
    centered_write()

    for line in art_lines:
        centered_write(line, delay=0.008)

    centered_write()
    separator()

    for line in info_lines:
        framed_write(line, delay=0.006)

    hline("┗", "┛")

    sys.stdout.write("\033[0m")
    sys.stdout.flush()
