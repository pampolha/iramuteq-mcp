"""Input bounds that are enforced before any R work starts."""

import pytest

from iramuteq_mcp.bridge import textops


@pytest.mark.parametrize("value", [50, 200, 50000])
def test_max_actives_accepts_range(value):
    assert textops.validate_max_actives(value) == value


@pytest.mark.parametrize("value", [49, 50001, 0, -1, True, "200", 1.5])
def test_max_actives_rejects_out_of_range_or_wrong_type(value):
    with pytest.raises(ValueError, match="max_actives"):
        textops.validate_max_actives(value)
