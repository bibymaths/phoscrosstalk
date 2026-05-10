import tomllib
from pathlib import Path
from phoscrosstalk.logger import get_logger

logger = get_logger()

MARKER_START = "# >>> auto-generated from config.toml [paths] >>>"
MARKER_END = "# <<< end auto-generated <<<"


def sync_gitignore(config_path: str = "config.toml", gitignore_path: str = ".gitignore"):
    config_path = Path(config_path)
    gitignore_path = Path(gitignore_path)

    # silently skip if no config found
    if not config_path.exists():
        return

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    paths = [v for v in config.get("paths", {}).values() if v]

    new_block = [MARKER_START] + paths + [MARKER_END]

    existing = gitignore_path.read_text().splitlines() if gitignore_path.exists() else []
    filtered, inside = [], False
    for line in existing:
        if line == MARKER_START:
            inside = True
        elif line == MARKER_END:
            inside = False
        elif not inside:
            filtered.append(line)

    final = filtered + [""] + new_block
    gitignore_path.write_text("\n".join(final) + "\n")
    logger.success(f"[*] '{gitignore_path.name}' synced from config.toml [paths]")
