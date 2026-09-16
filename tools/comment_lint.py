#!/usr/bin/env python3
"""Pre-submit detector for comment-hygiene violations in Rust sources.

Mirrors the deterministic audits the QA checker runs, so violations are
caught locally before a diff goes out for review. Stdlib only.

Checks (comment lines only, unless --all-lines):
  dangling   comment ends on a determiner/preposition/conjunction/negation
  paren      parenthetical not whole on one line (unbalanced in comment text)
  caps       ALL-CAPS word not in the allowlist (backticked code ignored)
  length     line longer than 128 columns
  artifact   review-artifact ID in code (RC-, QA-, SLOP-, BUG-, ESC-, ...)
  emdash     em/en dash as prose punctuation (house rule: restructure)
  theopener  /// doc comment opening with "The" (open with what it is)

Semicolons: the house skill prefers restructuring into bullets/arrows,
but repo precedent accepts them (never flagged in 8 QA rounds) and this
linter deliberately does not enforce that rule to limit churn.

Usage:
  python3 tools/comment_lint.py            # changed lines vs HEAD (default)
  python3 tools/comment_lint.py --all      # every comment line in the repo
  python3 tools/comment_lint.py src/foo.rs # specific paths, whole files

Exit 0 when clean, 1 when violations found.
"""
import argparse
import re
import subprocess
import sys

# Words a comment line must never end on (clauses continue after them).
# Extend freely; matching is case-insensitive on the final word.
DANGLERS = {
    "a", "an", "the", "this", "that", "these", "those", "its", "his", "her",
    "and", "or", "nor", "but", "yet", "so",
    "of", "for", "to", "in", "on", "at", "by", "as", "from", "with", "into",
    "onto", "over", "under", "after", "before", "below", "above", "between",
    "during", "through", "than", "per", "via",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "not", "no", "if", "when", "while", "then", "which", "whose", "like",
    "every", "each", "any", "all", "both",
}

# Uppercase tokens that are legitimate (units, protocols, hist terms...).
CAPS_ALLOW = {
    "AI", "ANSI", "API", "ASCII", "BYO", "CPU", "CRC", "C0", "C1", "CSI",
    "DEL", "E2E", "ESC", "HTTP", "ID", "ITL", "JSON", "KV", "L2", "LE",
    "LRU", "NaN", "PID", "PR", "RSS", "RUNNING", "TLS", "TODO", "TTL",
    "TUI", "TTFT", "UTF", "YAML", "P50", "P95", "P99", "FYI", "FIXME",
    "URL", "UI", "OK", "NO", "SSE",
}

ARTIFACT = re.compile(r"\b(?:RC|QA|SLOP|BUG|ESC|ARCH|CONS|COV|SEC|SPEC|"
                      r"MATH|RID|TASK|MISSION)[-/]?\d+\b|\bM\d{1,2}\b")

MAX_COLS = 128

# Settled-precedent lines previously audited and accepted by QA; never
# reopened. Extend as new precedent is ratified.
PRECEDENT = {
    "tests/integration.rs": {140, 200, 318},
    "src/derive.rs": {35},
}

BACKTICK = re.compile(r"`[^`]*`")
COMMENT = re.compile(r"^\s*//")
DOC_COMMENT = re.compile(r"^\s*///")
EMDASH = re.compile(r"—|(?<=\w)\s–\s(?=\w)")
THE_OPENER = re.compile(r"^The\b")
WORD_TAIL = re.compile(r"([A-Za-z0-9']+)$")
TRAILING = ".,;:)]}\"'…*"
CAPS_WORD = re.compile(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9]{1,}(?![A-Za-z0-9_])")
# Prometheus half-open interval notation `(a, b]` — not prose parens
INTERVAL = re.compile(r"\([^()\n]*\]")


def strip_trailing(text: str) -> str:
    text = text.rstrip()
    while text and text[-1] in TRAILING:
        text = text[:-1].rstrip()
    return text


def check_comment_block(path: str, start: int, lines):
    """Yield (lineno, rule, message) for a whole // or /// comment block.

    Block-level rules (theopener) need the block boundary; the per-line
    rules still run through check_line for every line of the block.
    """
    text = lines[0].strip().lstrip("/").strip()
    if DOC_COMMENT.match(lines[0]) and THE_OPENER.match(text):
        yield start, "theopener", 'doc comment opens with "The"'


def check_line(path: str, lineno: int, raw: str, all_lines: bool,
               fenced: bool = False):
    """Yield (lineno, rule, message) for one source line."""
    stripped = raw.rstrip("\n")
    if not all_lines and not COMMENT.match(stripped):
        return
    text = stripped.strip()
    if text.startswith("//"):
        text = text.lstrip("/").strip()

    # dangling clause ending
    clean = strip_trailing(text)
    m = WORD_TAIL.search(clean)
    if m and m.group(1).lower() in DANGLERS:
        yield lineno, "dangling", f'comment ends on "{m.group(1)}"'

    # prose semicolon at end of line (mid-line separators are accepted
    # per repo precedent, trailing ones are not; doctest code fences
    # are code, not prose)
    if text.endswith(";") and not fenced:
        yield lineno, "semicolon", "prose semicolon at end of line"

    # parenthetical must be whole on one line (backticked code ignored;
    # interval notation like `(0.1, 1]` is math, not prose parens)
    prose = BACKTICK.sub("", text)
    prose = INTERVAL.sub("", prose)
    if prose.count("(") != prose.count(")"):
        yield lineno, "paren", (
            f"parenthetical not whole on this line "
            f"({prose.count('(')} open / {prose.count(')')} close)")

    # stray ALL-CAPS words outside backticks
    for w in CAPS_WORD.findall(BACKTICK.sub("", text)):
        if w not in CAPS_ALLOW:
            yield lineno, "caps", f'uppercase "{w}" not in allowlist'

    # house rule: no dashes as prose punctuation
    if EMDASH.search(text):
        yield lineno, "emdash", "dash used as prose punctuation"

    # artifact IDs anywhere in the line (code or comment)
    for w in ARTIFACT.findall(stripped):
        yield lineno, "artifact", f'review-artifact id "{w}"'

    if len(stripped.expandtabs(4)) > MAX_COLS:
        yield lineno, "length", f"{len(stripped.expandtabs(4))} cols > {MAX_COLS}"


def git_changed_lines(paths):
    """Lines added vs HEAD per file (unified=0 diff), mirroring QA scope."""
    out = subprocess.run(
        ["git", "diff", "-U0", "HEAD", "--"] + paths,
        capture_output=True, text=True, timeout=60,
    ).stdout
    changed, cur = {}, None
    for ln in out.splitlines():
        if ln.startswith("+++ b/"):
            cur = ln[6:]
        elif ln.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
            if cur and m:
                start = int(m.group(1))
                count = int(m.group(2) or "1")
                changed.setdefault(cur, set()).update(
                    range(start, start + max(count, 1)))
        elif ln.startswith("+") and cur and not ln.startswith("+++"):
            pass  # positions come from hunk headers
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["src", "tests"])
    ap.add_argument("--all", action="store_true",
                    help="scan every comment line, not just changed ones")
    ap.add_argument("--all-lines", action="store_true",
                    help="also lint non-comment lines (length/artifact)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        pp = __import__("pathlib").Path(p)
        if pp.is_dir():
            files += sorted(str(f) for f in pp.rglob("*.rs"))
        elif pp.exists():
            files.append(str(pp))
    if not files:
        print("no Rust files matched", file=sys.stderr)
        return 2

    changed = {} if args.all else git_changed_lines(files)
    findings = 0
    for path in files:
        waivable = PRECEDENT.get(path, set()) if not args.all else set()
        want = changed.get(path)
        if want is None and not args.all:
            continue  # file untouched vs HEAD: outside review scope
        block_start = None  # first line number of the current comment block
        in_fence = False  # inside a ``` doctest fence: code, not prose
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, raw in enumerate(fh, 1):
                is_comment = bool(COMMENT.match(raw.rstrip("\n")))
                if is_comment and block_start is None:
                    block_start = i
                elif not is_comment:
                    if block_start is not None and (
                            want is None or block_start in want):
                        with open(path, encoding="utf-8") as fh2:
                            block = fh2.readlines()[block_start - 1:i - 1]
                        for lineno, rule, msg in check_comment_block(
                                path, block_start, block):
                            if lineno not in waivable:
                                print(f"{path}:{lineno}: [{rule}] {msg}")
                                findings += 1
                    block_start = None
                if i in waivable:
                    continue
                if want is not None and i not in want:
                    continue
                for lineno, rule, msg in check_line(
                        path, i, raw, args.all_lines, fenced=in_fence):
                    print(f"{path}:{lineno}: [{rule}] {msg}")
                    findings += 1
                if '```' in raw:
                    in_fence = not in_fence

    print(f"\n{findings} violation(s)"
          + ("" if args.all else " (changed lines vs HEAD only,"
             " pass --all for the full scan)"))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
