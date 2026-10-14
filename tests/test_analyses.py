"""Every analysis tool end-to-end against the fixture corpus and matrix (needs R)."""

import os
import shutil

import pytest
from helpers import run, wait_job

from iramuteq_mcp import (
    analysis_list,
    corpus_alceste,
    corpus_build,
    corpus_class_cloud,
    corpus_concordance,
    corpus_export_class,
    corpus_extract_mods,
    corpus_extract_themes,
    corpus_keys,
    corpus_labbe,
    corpus_lexico,
    corpus_similitude,
    corpus_split_by_variable,
    corpus_stats,
    corpus_tgen_prof,
    corpus_tgen_spec,
    corpus_validate,
    corpus_wordcloud,
    matrix_chi2,
    matrix_freqmulti,
    matrix_frequencies,
    matrix_mcnemar,
    matrix_proto,
    matrix_reinert,
    matrix_similitude,
    matrix_splitvar,
)
from iramuteq_mcp.bridge.jobs import JobStatus

pytestmark = pytest.mark.requires_r


@pytest.fixture(scope="module")
def alceste(corpus_path):
    job = run(corpus_alceste(str(corpus_path), lang="french", nbcl_p1=2, max_actives=200))
    res = wait_job(job["job_id"])
    assert res["status"] == JobStatus.DONE, res.get("error")
    return res["result"]


# ----------------------------------------------------------------- corpus


def test_keys_report_defaults():
    assert run(corpus_keys())["nom"] == "active"


def test_stats_and_key_override(corpus_path):
    assert run(corpus_stats(str(corpus_path), lang="french"))["counts"]["forms_total"] > 0
    assert run(corpus_stats(str(corpus_path), lang="french", active_keys=["adv"]))["counts"]


def test_unknown_key_override_is_rejected(corpus_path):
    job = run(corpus_lexico(str(corpus_path), lang="french", mineff=3, active_keys=["not_a_key"]))
    res = wait_job(job["job_id"])
    assert res["status"] == JobStatus.ERROR
    assert res["error_kind"] == "invalid_input"
    assert "not_a_key" in res["error"]


def test_concordance_finds_the_lemma(corpus_path):
    res = run(corpus_concordance(str(corpus_path), "travail", lang="french", limit=5))
    assert res["found"] and res["hits"]


def test_concordance_matches_inflected_forms(corpus_path):
    res = run(corpus_concordance(str(corpus_path), "compétence", lang="french", limit=5))
    assert res["found"]
    assert res["forms"] == ["compétences"]
    assert res["hits"], "the lemma literal never occurs: only its inflected form does"
    assert {h["form"] for h in res["hits"]} == {"compétences"}
    assert all(h["match"] for h in res["hits"])


def test_validate_reports_errors_reported(corpus_path):
    res = run(corpus_validate(str(corpus_path)))
    assert "errors_reported" in res
    assert "total_errors" not in res


def test_validate_accepts_variable_modality_and_plain_headers(tmp_path):
    p = tmp_path / "ok.txt"
    p.write_text(
        "**** *sexe_h *age_20\n-*travail\nTexte une.\n\n****\n-*travail\nTexte deux.\n",
        encoding="utf8",
    )
    assert run(corpus_validate(str(p)))["ok"]


def test_validate_rejects_equals_form_of_a_variable(tmp_path):
    p = tmp_path / "eq.txt"
    p.write_text("**** *QUESTAO=Q1\n-*q1\nTexte.\n", encoding="utf8")
    res = run(corpus_validate(str(p)))
    assert not res["ok"]
    assert "*QUESTAO_Q1" in res["errors"][0]["message"]


def test_validate_rejects_header_token_without_underscore(tmp_path):
    p = tmp_path / "no_underscore.txt"
    p.write_text("**** *QUESTAOQ1\n-*q1\nTexte.\n", encoding="utf8")
    res = run(corpus_validate(str(p)))
    assert not res["ok"]
    assert "variable_modality" in res["errors"][0]["message"]


def test_validate_hints_at_the_header_line_for_a_starred_line(tmp_path):
    p = tmp_path / "two_lines.txt"
    p.write_text("****\n*QUESTAO=Q1\n-*q1\nTexte.\n", encoding="utf8")
    res = run(corpus_validate(str(p)))
    assert not res["ok"]
    assert "*QUESTAO_Q1" in res["errors"][0]["message"]


def test_lexico_error_explains_the_expected_header_format(tmp_path):
    p = tmp_path / "bad_vars.txt"
    p.write_text("**** *QUESTAO=Q1\n-*q1\nTexte de la réponse.\n", encoding="utf8")
    job = run(corpus_lexico(str(p), lang="french"))
    res = wait_job(job["job_id"])
    assert res["status"] == JobStatus.ERROR, res.get("error")
    assert res["error_kind"] == "invalid_input"
    assert "*QUESTAO=Q1" in res["error"]
    assert "**** *sexe_h" in res["error"]


def _corpus_dirs(corpus_path):
    return {d for d in os.listdir(corpus_path.parent) if d.startswith(corpus_path.stem + "_corpus_")}


def test_corpus_build_cache_is_reused_by_analyses(corpus_path):
    build = run(corpus_build(str(corpus_path), lang="french"))
    before = _corpus_dirs(corpus_path)
    stats = run(corpus_stats(str(corpus_path), lang="french"))
    assert stats["corpus"]["corpus_uuid"] == build["corpus_uuid"]
    assert _corpus_dirs(corpus_path) == before, "corpus_stats rebuilt the corpus"


def test_segmentation_is_passed_from_build_to_analysis(corpus_path):
    build = run(corpus_build(str(corpus_path), lang="french", ucemethod=0, ucesize=60))
    before = _corpus_dirs(corpus_path)
    stats = run(corpus_stats(str(corpus_path), lang="french", ucemethod=0, ucesize=60))
    assert stats["corpus"]["corpus_uuid"] == build["corpus_uuid"]
    assert _corpus_dirs(corpus_path) == before, "corpus_stats ignored the built corpus"


def test_similitude_rejects_unknown_forms(corpus_path):
    job = run(corpus_similitude(str(corpus_path), lang="french", forms=["zzz_not_a_form"]))
    res = wait_job(job["job_id"])
    assert res["status"] == JobStatus.ERROR
    assert res["error_kind"] == "invalid_input"
    assert "zzz_not_a_form" in res["error"]


def test_alceste_produces_classes_and_profiles(alceste):
    assert alceste["class_sizes"]
    assert alceste["profiles"]
    assert os.path.isfile(os.path.join(alceste["output_dir"], "uce.csv"))


def test_labbe(corpus_path):
    res = wait_job(run(corpus_labbe(str(corpus_path), lang="french", mineff=3))["job_id"])
    assert res["status"] == JobStatus.DONE, res.get("error")
    assert res["result"]["distance_matrix"]


def test_lexico(corpus_path):
    res = wait_job(run(corpus_lexico(str(corpus_path), lang="french", mineff=3))["job_id"])
    assert res["status"] == JobStatus.DONE, res.get("error")
    assert res["result"]["specificities_forms"]


def test_wordcloud_and_subset(corpus_path):
    full = run(corpus_wordcloud(str(corpus_path), lang="french", mode=0, maxword=30, col_text=[10, 10, 80]))
    assert full["graph"] and os.path.isfile(full["graph"])
    subset = run(corpus_wordcloud(str(corpus_path), lang="french", mode=0, forms=["travail"]))
    assert subset["selected_count"] == 1


def test_class_cloud(corpus_path, alceste):
    by_freq = run(corpus_class_cloud(str(corpus_path), alceste["output_dir"], lang="french", class_id=1, indice=0))
    assert by_freq["graph"]
    by_chi2 = run(corpus_class_cloud(str(corpus_path), alceste["output_dir"], lang="french", class_id=2, indice=1))
    assert by_chi2["graph"]


def test_tgen_spec_filters_unknown_lemmas(corpus_path, tgen_path):
    res = run(corpus_tgen_spec(str(corpus_path), str(tgen_path), lang="french"))
    assert res["tables"]["specificities"]
    assert res["tgen"]["dropped_lemmas"], "fixture contains lemmas absent from the corpus"


def test_tgen_prof_job(corpus_path, tgen_path, alceste):
    res = wait_job(
        run(corpus_tgen_prof(str(corpus_path), alceste["output_dir"], str(tgen_path), lang="french"))["job_id"]
    )
    assert res["status"] == JobStatus.DONE, res.get("error")


def test_export_class_writes_file(corpus_path, alceste):
    res = run(corpus_export_class(str(corpus_path), alceste["output_dir"], lang="french", class_id=1))
    assert os.path.getsize(res["output"]) > 0


def test_subcorpus_tools_on_a_copy(corpus_path, tmp_path):
    copy = tmp_path / "mini_corpus.txt"
    shutil.copy(corpus_path, copy)
    split = run(corpus_split_by_variable(str(copy), "sexe"))
    assert split["count"] == 2
    themes = run(corpus_extract_themes(str(copy), ["travail"]))
    assert os.path.isfile(themes["output"])
    mods = run(corpus_extract_mods(str(copy), ["*sexe_h"], onefile=True))
    assert len(mods["created"]) == 1


def test_history_is_reachable():
    res = run(analysis_list())
    assert "corpora" in res
    assert "history_file" in res
    assert "jobs" not in res


# ----------------------------------------------------------------- matrix


def test_matrix_frequencies_and_freqmulti(matrix_path):
    assert run(matrix_frequencies(str(matrix_path), columns=[1, 2]))["tables"]
    assert run(matrix_freqmulti(str(matrix_path), columns=[4, 5]))["frequencies"]


def test_matrix_chi2_and_mcnemar(matrix_path):
    assert "results" in run(matrix_chi2(str(matrix_path), columns_a=[0], columns_b=[3]))
    assert "output_dir" in run(matrix_mcnemar(str(matrix_path), columns_a=[0], columns_b=[3]))


def test_matrix_splitvar(matrix_path):
    mods = run(matrix_splitvar(str(matrix_path), column=3))["modalities"]
    assert set(mods) == {"A", "B"} and all(mods.values())


def test_matrix_proto(matrix_path):
    res = run(matrix_proto(str(matrix_path), words_columns=[4, 5], rank_columns=[6, 7], freqmin=2))
    assert res["words"]


def test_matrix_reinert_and_similitude_jobs(matrix_path):
    res = wait_job(
        run(matrix_reinert(str(matrix_path), active_columns=[1, 2, 3], supplementary_columns=[0], nbcl_p1=3))["job_id"]
    )
    assert res["status"] == JobStatus.DONE, res.get("error")
    res = wait_job(run(matrix_similitude(str(matrix_path), columns=[1, 2, 3]))["job_id"])
    assert res["status"] == JobStatus.DONE, res.get("error")
