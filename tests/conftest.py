from pathlib import Path

import pytest

# Test data lives in the repo's examples/ (not versioned: *.parquet is
# git-ignored), so resolve it against the repo root rather than the cwd and
# skip — not error — the tests that need it when it is absent.
ROOT = Path(__file__).resolve().parent.parent
PTO_300K = ROOT / "examples" / "PbTiO3" / "pbtio3_222_300K.parquet"


@pytest.fixture(scope="session")
def pto_300k() -> Path:
    if not PTO_300K.is_file():
        pytest.skip(f"AIMD test data not found: {PTO_300K}")
    return PTO_300K
