"""Every analysis tool end-to-end against the fixture corpus and matrix (needs R)."""

import os
import shutil

import pytest
from helpers import run, wait_job

from iramuteq_mcp import (
    analysis_list,
    corpus_alceste,
    corpus_class_cloud,
    corpus_concordance,
    corpus_export_class,
    corpus_extract_mods,
    corpus_extract_themes,
    corpus_keys,
    corpus_labbe,
    corpus_lexico,
    corpus_split_by_variable,
    corpus_stats,
    corpus_tgen_prof,
    corpus_tgen_spec,
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

pytestmark = pytest.mark.requires_r


@pytest.fixture(scope="module")
def alceste(corpus_path):
    job = run(corpus_alceste(str(corpus_path), nbcl_p1=2, max_actives=200))
    res = wait_job(job["job_id"])
    assert res["status"] == "done", res.get("error")
    return res["result"]


# ----------------------------------------------------------------- corpus


def test_keys_report_defaults():
    assert run(corpus_keys())["nom"] == "active"


def test_stats_and_key_override(corpus_path):
    assert run(corpus_stats(str(corpus_path)))["counts"]["forms_total"] > 0
    assert run(corpus_stats(str(corpus_path), active_keys=["adv"]))["counts"]


def test_unknown_key_override_is_rejected(corpus_path):
    job = run(corpus_lexico(str(corpus_path), mineff=3, active_keys=["not_a_key"]))
    res = wait_job(job["job_id"])
    assert res["status"] == "error"
    assert res["error_kind"] == "invalid_input"
    assert "not_a_key" in res["error"]


def test_concordance_finds_the_lemma(corpus_path):
    res = run(corpus_concordance(str(corpus_path), "travail", limit=5))
    assert res["found"] and res["hits"]


def test_alceste_produces_classes_and_profiles(alceste):
    assert alceste["class_sizes"]
    assert alceste["profiles"]
    assert os.path.isfile(os.path.join(alceste["output_dir"], "uce.csv"))


def test_labbe(corpus_path):
    res = wait_job(run(corpus_labbe(str(corpus_path), mineff=3))["job_id"])
    assert res["status"] == "done", res.get("error")
    assert res["result"]["distance_matrix"]


def test_lexico(corpus_path):
    res = wait_job(run(corpus_lexico(str(corpus_path), mineff=3))["job_id"])
    assert res["status"] == "done", res.get("error")
    assert res["result"]["specificities_forms"]


def test_wordcloud_and_subset(corpus_path):
    full = run(corpus_wordcloud(str(corpus_path), mode=0, maxword=30, col_text=[10, 10, 80]))
    assert full["graph"] and os.path.isfile(full["graph"])
    subset = run(corpus_wordcloud(str(corpus_path), mode=0, forms=["travail"]))
    assert subset["selected_count"] == 1


def test_class_cloud(corpus_path, alceste):
    by_freq = run(corpus_class_cloud(str(corpus_path), alceste["output_dir"], class_id=1, indice=0))
    assert by_freq["graph"]
    by_chi2 = run(corpus_class_cloud(str(corpus_path), alceste["output_dir"], class_id=2, indice=1))
    assert by_chi2["graph"]


def test_tgen_spec_filters_unknown_lemmas(corpus_path, tgen_path):
    res = run(corpus_tgen_spec(str(corpus_path), str(tgen_path)))
    assert res["tables"]["specificities"]
    assert res["tgen"]["dropped_lemmas"], "fixture contains lemmas absent from the corpus"


def test_tgen_prof_job(corpus_path, tgen_path, alceste):
    res = wait_job(run(corpus_tgen_prof(str(corpus_path), alceste["output_dir"], str(tgen_path)))["job_id"])
    assert res["status"] == "done", res.get("error")


def test_export_class_writes_file(corpus_path, alceste):
    res = run(corpus_export_class(str(corpus_path), alceste["output_dir"], class_id=1))
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
    assert "corpora" in run(analysis_list())


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
    assert res["status"] == "done", res.get("error")
    res = wait_job(run(matrix_similitude(str(matrix_path), columns=[1, 2, 3]))["job_id"])
    assert res["status"] == "done", res.get("error")
