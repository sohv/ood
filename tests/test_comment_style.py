# enforces the comment rules in claude.md across every python file in src, scripts and tests.
# uv run -m pytest tests/test_comment_style.py -v

import re
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SOURCE_DIRS = ("src", "scripts", "tests")

# tooling directives and shebangs are not prose, so the lowercase rule does not apply to them
DIRECTIVES = ("!", "noqa", "type:", "ruff:", "fmt:", "pylint:", "pragma", "mypy:")
# capitalised python literals read correctly at the start of a comment
ALLOWED_CAPS = {"None", "True", "False"}
SEPARATOR = re.compile(r"^#\s*(.)\1{3,}\s*$")
BANNER = re.compile(r"""(["'])([#*=\-_+~.])\1\s*\*\s*\d+""")
IMPORT = re.compile(r"^(import \w|from [\w.]+ import )")
RUN_COMMAND = re.compile(r"^uv run ")
MAX_COMMENT_RUN = 1


def python_files() -> list[Path]:
    return sorted(path for name in SOURCE_DIRS for path in (REPO_ROOT / name).rglob("*.py"))


def comment_tokens(path: Path) -> list[tokenize.TokenInfo]:
    with path.open("rb") as handle:
        return [token for token in tokenize.tokenize(handle.readline) if token.type == tokenize.COMMENT]


def comment_body(token: tokenize.TokenInfo) -> str:
    return token.string.lstrip("#").strip()


def is_standalone(token: tokenize.TokenInfo) -> bool:
    return token.line.lstrip().startswith("#")


def location(path: Path, token: tokenize.TokenInfo) -> str:
    return f"{path.relative_to(REPO_ROOT)}:{token.start[0]}"


def header_tokens(tokens: list[tokenize.TokenInfo]) -> list[tokenize.TokenInfo]:
    if not tokens or tokens[0].start[0] != 1:
        return []
    header: list[tokenize.TokenInfo] = []
    for token in tokens:
        if not is_standalone(token) or (header and token.start[0] != header[-1].start[0] + 1):
            break
        header.append(token)
    return header


def test_source_files_are_discovered():
    # guards every test below from passing vacuously on an empty file list
    assert len(python_files()) > 5


def test_no_decorative_separators():
    violations = []
    for path in python_files():
        for token in comment_tokens(path):
            if SEPARATOR.match(token.string.strip()):
                violations.append(f"{location(path, token)}: {token.string.strip()}")
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if BANNER.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not violations, f"decorative separators are banned: {violations}"


def test_comments_start_lowercase():
    violations = []
    for path in python_files():
        for token in comment_tokens(path):
            body = comment_body(token)
            if not body or body.startswith(DIRECTIVES) or not body[0].isalpha():
                continue
            first_word = body.split()[0]
            if body[0].isupper() and first_word not in ALLOWED_CAPS and not first_word.isupper():
                violations.append(f"{location(path, token)}: {body}")
    assert not violations, f"comments start with a lowercase letter: {violations}"


def test_no_comment_paragraphs():
    violations = []
    for path in python_files():
        tokens = comment_tokens(path)
        # the header is a description line plus its run command, checked separately below
        header = {token.start[0] for token in header_tokens(tokens)}
        run: list[tokenize.TokenInfo] = []
        for token in tokens:
            if token.start[0] in header:
                continue
            if not is_standalone(token):
                run = []
                continue
            if run and token.start[0] != run[-1].start[0] + 1:
                run = []
            run.append(token)
            if len(run) > MAX_COMMENT_RUN:
                violations.append(location(path, run[0]))
                run = []
    assert not violations, f"rationale longer than {MAX_COMMENT_RUN} lines belongs in docs/: {sorted(set(violations))}"


def test_imports_are_not_commented():
    violations = []
    for path in python_files():
        lines = path.read_text().splitlines()
        tokens = comment_tokens(path)
        # the header's run command sits directly above the imports by design
        header = {token.start[0] for token in header_tokens(tokens)}
        for token in tokens:
            index = token.start[0] - 1
            if token.start[0] in header:
                continue
            if not is_standalone(token):
                if IMPORT.match(lines[index].strip()):
                    violations.append(location(path, token))
                continue
            following = next((line.strip() for line in lines[index + 1 :] if line.strip()), "")
            if IMPORT.match(following):
                violations.append(location(path, token))
    assert not violations, f"imports do not get comments: {violations}"


def test_file_header_is_one_line_plus_run_command():
    violations = []
    for path in python_files():
        header = header_tokens(comment_tokens(path))
        described = [token for token in header if not RUN_COMMAND.match(comment_body(token))]
        if len(described) > 1:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {len(described)} description lines")
    assert not violations, f"a file header is one description line plus the run command: {violations}"
