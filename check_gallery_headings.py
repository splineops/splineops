#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import re
from typing import List, Tuple

Underline = Tuple[int, str, str, str]  # (lineno, char, title, underline)

def iter_text_blocks(lines: List[str]) -> List[Tuple[int, List[str]]]:
    """
    Return a list of (start_lineno, block_lines) for Sphinx-Gallery text blocks:
    contiguous lines starting with '# ' (hash + space).
    """
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].startswith("# "):
            start = i
            cur = []
            while i < n and lines[i].startswith("# "):
                cur.append(lines[i])
                i += 1
            blocks.append((start + 1, cur))  # 1-based line numbers
        else:
            i += 1
    return blocks

def find_headings(block: List[str], start_lineno: int) -> List[Underline]:
    """
    In a text block (already stripped to '# ' lines), find reST headings of the form:
      # Title
      # -----
    Return a list of (lineno_of_underline, underline_char, title, underline_text).
    """
    hits: List[Underline] = []
    for idx in range(len(block) - 1):
        title_line = block[idx][2:].rstrip("\n")        # drop '# '
        ul_line    = block[idx+1][2:].rstrip("\n")      # drop '# '
        if not title_line or not ul_line:
            continue
        # underline must be same char repeated (ignoring trailing spaces)
        ul_stripped = ul_line.strip()
        if not ul_stripped:
            continue
        if len(set(ul_stripped)) == 1:
            ch = ul_stripped[0]
            # sanity: typical adornment chars (docutils allows many; this is just a hint)
            if ch in "= -~^*+#:\"'`.,":
                hits.append((start_lineno + idx + 1, ch, title_line, ul_line))
    return hits

def main():
    ap = argparse.ArgumentParser(description="Check reST headings in Sphinx-Gallery text blocks (# ... lines). No changes made.")
    ap.add_argument("path", help="Python example file (.py)")
    args = ap.parse_args()

    p = Path(args.path)
    text = p.read_text(encoding="utf-8")

    lines = text.splitlines()
    blocks = iter_text_blocks(lines)

    all_headings: List[Underline] = []
    for start_lineno, block in blocks:
        all_headings.extend(find_headings(block, start_lineno))

    if not all_headings:
        print("No headings found in # ... text blocks.")
        return

    print(f"Found {len(all_headings)} headings:\n")
    adornment_order: List[str] = []
    problems = 0

    for lineno, ch, title, underline in all_headings:
        ul = underline[2:]  # original underline text after '# '
        ul_stripped = ul.strip()
        ok_repeat = (len(set(ul_stripped)) == 1)
        ok_len = (len(ul_stripped) >= len(title))
        ok_trim = (ul == ul.strip())  # no leading/trailing spaces

        if ch not in adornment_order:
            adornment_order.append(ch)

        status = []
        if not ok_repeat:
            status.append("underline not a single repeated char")
        if not ok_len:
            status.append(f"underline too short ({len(ul_stripped)} < {len(title)})")
        if not ok_trim:
            status.append("underline has leading/trailing spaces")
        flag = "OK" if not status else "PROBLEM"
        if status:
            problems += 1

        print(f"{p}:{lineno}: “{title}”  underline ‘{ch}’  -> {flag}")
        if status:
            for s in status:
                print(f"    - {s}")
        # Show counts to eyeball mismatches quickly
        print(f"    title len={len(title)}, underline len={len(ul_stripped)}")

    print("\nAdornment (underline char) order in this file (defines levels):", "  ".join(adornment_order))
    if problems:
        print(f"\n{problems} problem(s) detected. Fix the lines above (usually make the underline longer and strip spaces).")
    else:
        print("\nAll headings look structurally valid.")

if __name__ == "__main__":
    main()
