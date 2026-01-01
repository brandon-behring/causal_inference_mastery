#!/usr/bin/env python3
"""
Automated Metrics Generator for causal_inference_mastery.

Generates accurate metrics for documentation. Run after major changes
to keep CLAUDE.md, README.md, and METRICS_CURRENT.md consistent.

Usage:
    python scripts/update_metrics.py              # Print to stdout
    python scripts/update_metrics.py --output     # Write to docs/METRICS_CURRENT.md

Session 158: Created as part of documentation automation.
"""

import argparse
import subprocess
import re
from datetime import datetime
from pathlib import Path


def count_python_tests(tests_dir: Path) -> int:
    """Count Python test functions (def test_*)."""
    result = subprocess.run(
        ["grep", "-r", "-E", r"^\s*def test_", str(tests_dir)],
        capture_output=True,
        text=True,
    )
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0


def count_julia_assertions(julia_test_dir: Path) -> int:
    """Count Julia @test assertions."""
    result = subprocess.run(
        ["grep", "-r", "-E", r"@test\s", str(julia_test_dir)],
        capture_output=True,
        text=True,
    )
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0


def count_source_lines(directory: Path, extension: str) -> int:
    """Count non-empty source lines for given extension."""
    total = 0
    for file in directory.rglob(f"*{extension}"):
        if "__pycache__" in str(file) or ".git" in str(file):
            continue
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line for line in f if line.strip()]
                total += len(lines)
        except Exception:
            continue
    return total


def count_method_families(src_dir: Path) -> list:
    """Identify method family directories in src/causal_inference/."""
    method_dirs = []
    ci_dir = src_dir / "causal_inference"
    if ci_dir.exists():
        for item in sorted(ci_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("_"):
                # Check if it has actual implementation files
                py_files = list(item.glob("*.py"))
                if py_files and not all(f.name == "__init__.py" for f in py_files):
                    method_dirs.append(item.name)
    return method_dirs


def get_current_session(current_work_path: Path) -> int:
    """Extract current session number from CURRENT_WORK.md."""
    if not current_work_path.exists():
        return 0
    content = current_work_path.read_text()
    match = re.search(r"Session\s*(\d+)", content, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def run_pytest_collect(tests_dir: Path) -> tuple:
    """Run pytest --collect-only and parse results."""
    result = subprocess.run(
        ["pytest", str(tests_dir), "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=tests_dir.parent,
    )
    # Parse "X tests collected" from output
    match = re.search(r"(\d+)\s+tests?\s+collected", result.stdout + result.stderr)
    collected = int(match.group(1)) if match else 0

    # Count errors
    errors = result.stderr.count("ERROR") + result.stdout.count("ERROR collecting")

    return collected, errors


def main():
    parser = argparse.ArgumentParser(description="Generate repository metrics")
    parser.add_argument("--output", action="store_true", help="Write to docs/METRICS_CURRENT.md")
    args = parser.parse_args()

    # Paths
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"
    tests_dir = repo_root / "tests"
    julia_src = repo_root / "julia" / "src"
    julia_test = repo_root / "julia" / "test"
    current_work = repo_root / "CURRENT_WORK.md"
    output_path = repo_root / "docs" / "METRICS_CURRENT.md"

    # Gather metrics
    print("Gathering metrics...", flush=True)

    python_tests = count_python_tests(tests_dir)
    julia_assertions = count_julia_assertions(julia_test)
    python_src_lines = count_source_lines(src_dir, ".py")
    julia_src_lines = count_source_lines(julia_src, ".jl")
    method_families = count_method_families(src_dir)
    current_session = get_current_session(current_work)

    # Pytest collection check
    print("Running pytest collection check...", flush=True)
    collected, collection_errors = run_pytest_collect(tests_dir)

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# Repository Metrics

**Generated**: {timestamp}
**Session**: {current_session}
**Generator**: `scripts/update_metrics.py`

---

## Test Counts

| Metric | Count |
|--------|-------|
| Python test functions (`def test_*`) | {python_tests:,} |
| Julia test assertions (`@test`) | {julia_assertions:,} |
| Total test assertions | {python_tests + julia_assertions:,} |
| Pytest collected (actual runnable) | {collected:,} |
| Collection errors | {collection_errors} |

---

## Source Line Counts

| Language | Directory | Lines (non-empty) |
|----------|-----------|-------------------|
| Python | `src/` | {python_src_lines:,} |
| Julia | `julia/src/` | {julia_src_lines:,} |
| **Total** | | **{python_src_lines + julia_src_lines:,}** |

---

## Method Families ({len(method_families)})

{chr(10).join(f"- `{m}`" for m in method_families)}

---

## Methodology

- **Python tests**: `grep -rE "^\\s*def test_" tests/`
- **Julia assertions**: `grep -rE "@test\\s" julia/test/`
- **Source lines**: Non-empty lines in `.py` and `.jl` files
- **Method families**: Directories in `src/causal_inference/` with implementation files

---

## Usage

Regenerate this file:
```bash
python scripts/update_metrics.py --output
```

Check metrics without writing:
```bash
python scripts/update_metrics.py
```
"""

    if args.output:
        output_path.write_text(report)
        print(f"Wrote metrics to {output_path}")
    else:
        print(report)

    # Summary for quick reference
    print("\n" + "=" * 50)
    print("SUMMARY (for copy-paste to CLAUDE.md):")
    print("=" * 50)
    print(f"Python tests: {python_tests:,}")
    print(f"Julia assertions: {julia_assertions:,}")
    print(f"Python src lines: {python_src_lines:,}")
    print(f"Julia src lines: {julia_src_lines:,}")
    print(f"Total lines: {python_src_lines + julia_src_lines:,}")
    print(f"Method families: {len(method_families)}")
    print(f"Collection errors: {collection_errors}")


if __name__ == "__main__":
    main()
