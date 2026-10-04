"""Data validation gate.

Run this before training (and wire it into CI / the Airflow DAG as its own
task) to check that a metadata CSV matches the expected schema. On a
violation it prints every failing check with the offending rows, writes a
machine-readable error report, and exits non-zero -- which is what makes
"the pipeline stops and alerts on bad data" true rather than aspirational:
any runner (shell, GitHub Actions, Airflow) treats a non-zero exit as a
failed step and stops the DAG / fails the CI job / fails the alert check.

Usage
-----
    python -m src.validation.validate_metadata data/raw/images/metadata.csv
    python -m src.validation.validate_metadata tests/fixtures/bad_metadata_sample.csv

Exit code 0  -> data is valid, summary printed.
Exit code 1  -> schema violation, details printed + saved to
                artifacts/validation/<input_stem>_errors.json
Exit code 2  -> input file missing / unreadable.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from pandera.errors import SchemaErrors

from src.validation.schema import metadata_schema

DEFAULT_PATH = Path("data/raw/images/metadata.csv")
REPORT_DIR = Path("artifacts/validation")


def validate(path: Path) -> int:
    if not path.exists():
        print(f"[FAIL] input file not found: {path}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - surfacing any parse failure is the point
        print(f"[FAIL] could not read {path}: {exc}", file=sys.stderr)
        return 2

    try:
        metadata_schema.validate(df, lazy=True)
    except SchemaErrors as err:
        failures = err.failure_cases
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"{path.stem}_errors.json"
        report_path.write_text(failures.to_json(orient="records", indent=2), encoding="utf-8")

        print(f"[FAIL] schema validation failed for {path}", file=sys.stderr)
        print(f"        {len(failures)} failing check(s) across "
              f"{failures['index'].nunique()} row(s)", file=sys.stderr)
        by_check = failures.groupby("check")["index"].apply(list)
        for check, rows in by_check.items():
            shown = rows[:5]
            more = f" (+{len(rows) - 5} more)" if len(rows) > 5 else ""
            print(f"        - {check}: rows {shown}{more}", file=sys.stderr)
        print(f"        full report: {report_path}", file=sys.stderr)
        return 1

    print(f"[PASS] {path}: {len(df)} rows, {len(df.columns)} columns, schema OK")
    return 0


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    sys.exit(validate(arg))
