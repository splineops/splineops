#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import unicodedata


def iter_files(paths, exts):
    """Yield files from given paths; if a path is a directory, recurse and filter by extensions."""
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and (not exts or f.suffix.lower() in exts):
                    yield f
        elif p.is_file():
            yield p
        else:
            print(f"warn: {p} not found", file=sys.stderr)


def scan_file(path: Path, encoding: str = "utf-8"):
    """Return a list of (lineno, colno, char, line_text) for non-ASCII chars in the file."""
    try:
        text = path.read_text(encoding=encoding, errors="strict")
    except UnicodeDecodeError:
        # Fall back so we can still inspect problematic files without crashing
        text = path.read_text(encoding=encoding, errors="replace")

    hits = []
    for lineno, line in enumerate(text.splitlines(keepends=False), 1):
        for colno, ch in enumerate(line, 1):
            if ord(ch) > 127:
                hits.append((lineno, colno, ch, line))
    return hits


def main():
    ap = argparse.ArgumentParser(
        description="Find non-ASCII characters in files (no modifications)."
    )
    ap.add_argument("paths", nargs="+", help="Files and/or directories to scan")
    ap.add_argument(
        "--ext",
        action="append",
        default=[".py"],
        help="Extension to include when scanning directories (repeatable). Default: .py",
    )
    ap.add_argument(
        "--encoding", default="utf-8", help="File encoding to read (default: utf-8)"
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Only print a summary instead of every occurrence",
    )
    args = ap.parse_args()

    exts = {e.lower() for e in args.ext}
    total = 0
    unique = set()

    for f in iter_files(args.paths, exts):
        hits = scan_file(f, args.encoding)
        for lineno, colno, ch, line in hits:
            total += 1
            unique.add(ord(ch))
            if not args.summary:
                name = unicodedata.name(ch, "UNKNOWN")
                cp = f"U+{ord(ch):04X}"
                # Make tabs visible so columns are less surprising when reviewing
                line_vis = line.replace("\t", "\\t")
                print(f"{f}:{lineno}:{colno}: {repr(ch)} ({cp} {name})")
                print(f"    {line_vis}")

    if total:
        cps = ", ".join(f"U+{cp:04X}" for cp in sorted(unique))
        print(f"\n{total} non-ASCII character(s) found in {len(unique)} unique code point(s): {cps}")
        sys.exit(1)  # non-zero so CI can fail if desired
    else:
        print("No non-ASCII characters found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
