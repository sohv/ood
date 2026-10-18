import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
# rules moved to docs/tooling.md are still rules, so both files are checked as one
CLAUDE_MD = (REPO_ROOT / "CLAUDE.md").read_text() + "\n" + (REPO_ROOT / "docs" / "tooling.md").read_text()
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text()
# pointers wrap across lines, so match against a whitespace-normalised copy
FLAT = " ".join(CLAUDE_MD.split())

# gitignored, created at runtime by the first cached llm call, so it can never ship in the template
RUNTIME_DIRS = {Path("cache")}


def documented_directories() -> list[Path]:
    block = re.search(r"```\nmy-project/\n(.*?)\n```", CLAUDE_MD, re.DOTALL).group(1)
    stack: list[str] = []
    paths = []
    for line in block.splitlines():
        match = re.match(r"^([│ ]*)[├└]── ([\w.\-]+/?)", line)
        if not match:
            continue
        depth = len(match.group(1)) // 4
        name = match.group(2)
        stack[depth:] = [name.rstrip("/")]
        if name.endswith("/"):
            paths.append(Path(*stack))
    return paths


def documented_modules() -> list[str]:
    return re.findall(r"`(src/[\w/]+\.py)`", FLAT)


def documented_symbols() -> list[tuple[str, str]]:
    # `setup_logging(output_dir)` from `src/utils/logging.py`
    calls = re.findall(r"`(\w+)\([^`]*\)` (?:from|in) `(src/[\w/]+\.py)`", FLAT)
    # the base `Config` in `src/utils/config.py`
    names = re.findall(r"`(\w+)` in `(src/[\w/]+\.py)`", FLAT)
    # from src.generation.cache import cached_llm_call
    imports = [
        (symbol, module.replace(".", "/") + ".py")
        for module, symbol in re.findall(r"from (src\.[\w.]+) import (\w+)", FLAT)
    ]
    return sorted(set(calls + names + imports))


def test_tree_block_is_parseable():
    # guards the tests below from silently passing on an empty match
    assert len(documented_directories()) > 5


def test_pointer_patterns_still_match():
    # if these regexes stop matching, the tests below pass vacuously on a pointer to a missing file
    assert len(documented_modules()) >= 4
    assert len(documented_symbols()) >= 4


def test_every_module_named_in_claude_md_exists():
    missing = [m for m in documented_modules() if not (REPO_ROOT / m).is_file()]
    assert not missing, f"CLAUDE.md points at modules that do not exist: {missing}"


def test_every_symbol_named_in_claude_md_is_defined():
    undefined = []
    for symbol, module in documented_symbols():
        path = REPO_ROOT / module
        if not path.is_file():
            undefined.append(f"{module} (missing file)")
            continue
        if not re.search(rf"^(?:async def|def|class) {symbol}\b", path.read_text(), re.M):
            undefined.append(f"{symbol} in {module}")
    assert not undefined, f"CLAUDE.md names symbols that are not defined: {undefined}"


def test_every_directory_in_the_tree_exists():
    documented = [p for p in documented_directories() if p not in RUNTIME_DIRS]
    missing = [str(p) for p in documented if not (REPO_ROOT / p).is_dir()]
    assert not missing, f"CLAUDE.md documents directories that do not exist: {missing}"


def test_ruff_policy_matches_pyproject():
    documented = re.search(r"line length (\d+), ruff ignores ([\w, ]+)\.", CLAUDE_MD)
    line_length = int(re.search(r"^line-length = (\d+)$", PYPROJECT, re.M).group(1))
    ignores = re.findall(r'"(\w+)"', re.search(r"^ignore = \[(.*)\]$", PYPROJECT, re.M).group(1))
    assert int(documented.group(1)) == line_length
    assert [code.strip() for code in documented.group(2).split(",")] == ignores
