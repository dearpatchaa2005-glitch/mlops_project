import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.validation.schema import metadata_schema

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_good_fixture_passes_schema():
    df = pd.read_csv(FIXTURES / "good_metadata_sample.csv")
    metadata_schema.validate(df, lazy=True)  # must not raise


def test_bad_fixture_fails_schema_with_every_injected_error_caught():
    df = pd.read_csv(FIXTURES / "bad_metadata_sample.csv")
    with pytest.raises(SchemaErrors) as excinfo:
        metadata_schema.validate(df, lazy=True)
    failing_checks = set(excinfo.value.failure_cases["check"])
    # one assertion per kind of corruption injected when the fixture was built
    assert any("isin" in c for c in failing_checks)          # bad category (sex="alien")
    assert any("in_range" in c for c in failing_checks)      # out-of-range eccentricity
    assert any("not_nullable" in c for c in failing_checks)  # missing patient_id
    assert any("str_matches" in c for c in failing_checks)   # malformed isic_id
    assert any("uniqueness" in c for c in failing_checks)    # duplicate isic_id
