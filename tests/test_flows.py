"""Flow tests: every analysis tool end-to-end (jobs polled to completion)."""

import asyncio
import json
import os
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, "src")

from iramuteq_mcp import (  # noqa: E402
    analysis_list,
    corpus_alceste,
    corpus_class_cloud,
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
    job_result,
    job_status,
    matrix_chi2,
    matrix_freqmulti,
    matrix_frequencies,
    matrix_mcnemar,
    matrix_open,
    matrix_proto,
    matrix_reinert,
    matrix_similitude,
    matrix_splitvar,
    status,
)

FAILURES = []


def run(coro):
    return asyncio.run(coro)


def wait_job(job_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = run(job_status(job_id))
        if st["status"] not in ("queued", "running"):
            return run(job_result(job_id))
        time.sleep(1)
    raise TimeoutError(job_id)


def check(label, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(label)


def make_matrix(path):
    random.seed(42)
    sexes = ["h", "f"]
    ages = ["18-25", "26-35", "36-50"]
    themes = ["sport", "musique", "lecture", "cinema"]
    words = ["ecole", "travail", "famille", "argent", "sante", "voyage"]
    with open(path, "w", encoding="utf8") as f:
        f.write("id;sex;age;theme;vote;mot1;mot2;rang1;rang2\n")
        for i in range(60):
            sex = sexes[i % 2]
            age = ages[i % 3]
            theme = themes[(i + i // 3) % 4]
            vote = "A" if (i % 2 == 0) == (theme in ("sport", "cinema")) else "B"
            m1, m2 = random.sample(words, 2)
            r1, r2 = random.randint(1, 20), random.randint(1, 20)
            f.write(f"m{i};{sex};{age};{theme};{vote};{m1};{m2};{r1};{r2}\n")


def main():
    st = run(status())
    check("status ready (R packages present)", st.get("ready") is True, str(st.get("r_packages_missing")))

    corpus = "tests/fixtures/mini_corpus.txt"
    tgen = "tests/fixtures/mini_tgen.txt"
    mpath = "tests/fixtures/mini_matrix.csv"
    make_matrix(mpath)

    # ---------------------------------------------------------------- corpus
    keys = run(corpus_keys())
    check("corpus_keys reports nom=active", keys.get("nom") == "active", str(keys.get("nom")))

    stats = run(corpus_stats(corpus))
    check("corpus_stats", bool(stats.get("counts")), str(stats.get("counts")))
    stats_keys = run(corpus_stats(corpus, active_keys=["adv"]))
    check("corpus_stats with key override", bool(stats_keys.get("counts")))

    # ALCESTE (job) — also the basis for class-level tools
    job = run(corpus_alceste(corpus, nbcl_p1=2, max_actives=200))
    res = wait_job(job["job_id"])
    alceste = res.get("result") or {}
    check("corpus_alceste done", res["status"] == "done", res.get("error") or "")
    alceste_dir = alceste.get("output_dir")

    # Labbé (job)
    job = run(corpus_labbe(corpus, mineff=3))
    res = wait_job(job["job_id"])
    check("corpus_labbe done", res["status"] == "done", (res.get("error") or "")[:300])
    if res["status"] == "done":
        check("labbe has distance matrix", bool((res["result"] or {}).get("distance_matrix")))

    lex_job = run(corpus_lexico(corpus, mineff=3))
    res = wait_job(lex_job["job_id"])
    check("corpus_lexico done", res["status"] == "done", (res.get("error") or "")[:300])

    # wordclouds (sync)
    wc = run(corpus_wordcloud(corpus, mode=0, maxword=30, col_text=[10, 10, 80]))
    check("corpus_wordcloud graph written", bool(wc.get("graph")) and os.path.isfile(wc["graph"]), str(wc.get("graph")))
    wc_forms = run(corpus_wordcloud(corpus, mode=0, forms=["travail"]))
    check("corpus_wordcloud forms subset", wc_forms.get("selected_count") == 1, str(wc_forms.get("selected_count")))
    if alceste_dir:
        cc = run(corpus_class_cloud(corpus, alceste_dir, class_id=1, indice=0))
        check("corpus_class_cloud graph", bool(cc.get("graph")), str(cc.get("graph")))
        cc2 = run(corpus_class_cloud(corpus, alceste_dir, class_id=2, indice=1))
        check("corpus_class_cloud chi2 weighting", bool(cc2.get("graph")))

    # Tgen (spec sync, prof job)
    spec = run(corpus_tgen_spec(corpus, tgen))
    check("corpus_tgen_spec tables", bool(spec.get("tables", {}).get("specificities")), str(spec.get("tgen")))
    if alceste_dir:
        job = run(corpus_tgen_prof(corpus, alceste_dir, tgen))
        res = wait_job(job["job_id"])
        check("corpus_tgen_prof done", res["status"] == "done", (res.get("error") or "")[:300])

        exp = run(corpus_export_class(corpus, alceste_dir, class_id=1))
        check("corpus_export_class writes file", os.path.isfile(exp.get("output", "")), str(exp.get("output")))

    # subcorpus tools operate on a throw-away copy (they write beside input)
    tmp = tempfile.mkdtemp(prefix="sub-")
    try:
        copy = os.path.join(tmp, "mini_corpus.txt")
        shutil.copy(corpus, copy)
        sv = run(corpus_split_by_variable(copy, "sexe"))
        check("corpus_split_by_variable", sv.get("count", 0) == 2, str(sv.get("created")))
        th = run(corpus_extract_themes(copy, ["travail"]))
        check("corpus_extract_themes", os.path.isfile(th.get("output", "")), str(th.get("output")))
        md = run(corpus_extract_mods(copy, ["*sexe_h"], onefile=True))
        check("corpus_extract_mods", len(md.get("created", [])) == 1, str(md.get("created")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---------------------------------------------------------------- matrix
    run(matrix_open(mpath))
    fr = run(matrix_frequencies(mpath, columns=[1, 2]))
    check("matrix_frequencies", bool(fr.get("tables")))
    fm = run(matrix_freqmulti(mpath, columns=[4, 5]))
    check("matrix_freqmulti", bool(fm.get("frequencies")), str(len(fm.get("frequencies", []))))
    chi = run(matrix_chi2(mpath, columns_a=[0], columns_b=[3]))
    check("matrix_chi2", chi.get("analysis") in ("chi2", "chi2mcnemar", None) or "results" in chi, str(list(chi)[:6]))
    mc = run(matrix_mcnemar(mpath, columns_a=[0], columns_b=[3]))
    check("matrix_mcnemar", "output_dir" in mc, str(mc.get("error", ""))[:200])
    sv = run(matrix_splitvar(mpath, column=3))
    mods = sv.get("modalities", {})
    check("matrix_splitvar creates one matrix per modality", len(mods) == 2 and all(mods.values()), str(mods))
    pr = run(matrix_proto(mpath, words_columns=[4, 5], rank_columns=[6, 7], freqmin=2))
    check("matrix_proto words", bool(pr.get("words")), str(len(pr.get("words", []))))

    job = run(matrix_reinert(mpath, active_columns=[1, 2, 3], supplementary_columns=[0], nbcl_p1=3))
    res = wait_job(job["job_id"])
    check("matrix_reinert done", res["status"] == "done", (res.get("error") or "")[:200])
    job = run(matrix_similitude(mpath, columns=[1, 2, 3]))
    res = wait_job(job["job_id"])
    check("matrix_similitude done", res["status"] == "done", (res.get("error") or "")[:200])

    hist = run(analysis_list())
    check("analysis_list reachable", "corpora" in hist)

    print("\n" + ("ALL OK" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    if FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
