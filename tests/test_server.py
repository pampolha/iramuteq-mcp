"""The MCP surface itself: tool registration and status (no R needed for registration)."""

import asyncio

import pytest
from fastmcp import Client
from helpers import run

from iramuteq_mcp import mcp, status

EXPECTED_TOOLS = {
    "status",
    "corpus_validate",
    "corpus_build",
    "corpus_stats",
    "corpus_concordance",
    "corpus_keys",
    "corpus_alceste",
    "corpus_lexico",
    "corpus_labbe",
    "corpus_similitude",
    "corpus_wordcloud",
    "corpus_class_cloud",
    "corpus_tgen_spec",
    "corpus_tgen_prof",
    "corpus_split_by_variable",
    "corpus_extract_themes",
    "corpus_extract_mods",
    "corpus_export_class",
    "matrix_open",
    "matrix_frequencies",
    "matrix_freqmulti",
    "matrix_chi2",
    "matrix_mcnemar",
    "matrix_splitvar",
    "matrix_proto",
    "matrix_reinert",
    "matrix_similitude",
    "job_status",
    "job_result",
    "job_cancel",
    "job_list",
    "job_prune",
    "analysis_read",
    "analysis_list",
}


def test_registered_tool_surface():
    async def names():
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert asyncio.run(names()) == EXPECTED_TOOLS


def test_unknown_job_id_is_a_tool_error():
    from fastmcp.exceptions import ToolError

    from iramuteq_mcp import job_cancel, job_result, job_status

    for tool in (job_status, job_result, job_cancel):
        with pytest.raises(ToolError, match="unknown job id"):
            run(tool("does-not-exist"))


def test_job_tools_validate_before_submitting_a_job():
    from fastmcp.exceptions import ToolError

    from iramuteq_mcp import corpus_labbe, matrix_reinert

    with pytest.raises(ToolError, match="indice"):
        run(corpus_labbe("/nonexistent/corpus.txt", indice="wrong"))
    with pytest.raises(ToolError, match="active_columns"):
        run(matrix_reinert("/nonexistent/table.csv"))
    with pytest.raises(ToolError, match="supplementary_columns"):
        run(matrix_reinert("/nonexistent/table.csv", active_columns=[0], supplementary_columns=[]))


@pytest.mark.requires_r
def test_status_reports_ready():
    st = run(status())
    assert st["ready"] is True
    assert st["r_packages_missing"] == []


def test_job_status_schema_hides_the_context_parameter():
    async def schemas():
        async with Client(mcp) as client:
            return {t.name: t.input_schema for t in await client.list_tools()}

    schema = asyncio.run(schemas())["job_status"]
    assert set(schema["properties"]) == {"job_id"}
    assert schema["required"] == ["job_id"]


def test_unknown_job_error_reaches_the_client_through_the_protocol():
    from fastmcp.exceptions import ToolError

    async def call():
        async with Client(mcp) as client:
            await client.call_tool("job_status", {"job_id": "does-not-exist"})

    with pytest.raises(ToolError, match="unknown job id"):
        asyncio.run(call())
