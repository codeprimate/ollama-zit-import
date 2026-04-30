"""CLI module entrypoint."""

import sys

from ollama_zit_import.cli import run


def main() -> None:
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
