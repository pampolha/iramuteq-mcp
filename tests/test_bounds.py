"""Input bounds that are enforced before any R work starts."""

import pytest
from fastmcp.exceptions import ToolError
from helpers import run

from iramuteq_mcp import corpus_similitude, corpus_wordcloud, matrix_similitude
from iramuteq_mcp.bridge import textops


@pytest.mark.parametrize("value", [50, 200, 50000])
def test_max_actives_accepts_range(value):
    assert textops.validate_max_actives(value) == value


@pytest.mark.parametrize("value", [49, 50001, 0, -1, True, "200", 1.5])
def test_max_actives_rejects_out_of_range_or_wrong_type(value):
    with pytest.raises(ValueError, match="max_actives"):
        textops.validate_max_actives(value)


@pytest.mark.parametrize("value", [8, -1, True, 2.5])
def test_similitude_rejects_unknown_community_algorithm(value):
    with pytest.raises(ToolError, match="communities"):
        run(corpus_similitude("/nonexistent/corpus.txt", lang="french", communities=value))
    with pytest.raises(ToolError, match="communities"):
        run(matrix_similitude("/nonexistent/table.csv", columns=[0], communities=value))


def test_wordcloud_rejects_forms_and_exclude_forms_together():
    with pytest.raises(ToolError, match="mutually exclusive"):
        run(corpus_wordcloud("/nonexistent/corpus.txt", lang="french", forms=["a"], exclude_forms=["b"]))
