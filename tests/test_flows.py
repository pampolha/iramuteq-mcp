"""Flow tests: job-based ALCESTE + matrix analyses end-to-end."""

import json
import sys
import time

sys.path.insert(0, "src")

from iramuteq_mcp import (  # noqa: E402
    analysis_list,
    analysis_read,
    corpus_alceste,
    corpus_lexico,
    job_list,
    job_result,
    job_status,
    matrix_chi2,
    matrix_frequencies,
    matrix_open,
    matrix_reinert,
    matrix_similitude,
    status,
)


def wait_job(job_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = asyncio_run(job_status(job_id))
        if st["status"] not in ("queued", "running"):
            return asyncio_run(job_result(job_id))
        time.sleep(1)
    raise TimeoutError(job_id)


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def pp(label, obj):
    print(f"\n===== {label} =====")
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False)[:1500])


def main():
    pp("status", asyncio_run(status()))

    corpus = "tests/fixtures/mini_corpus.txt"

    job = asyncio_run(corpus_alceste(corpus, nbcl_p1=2, max_actives=200))
    pp("alceste job submitted", job)
    res = wait_job(job["job_id"])
    print("alceste status:", res["status"], "| classes:", (res.get("result") or {}).get("class_sizes"))
    print("profiles keys:", list((res.get("result") or {}).get("profiles", {}).keys()))
    print("profile 1 top:", json.dumps((res.get("result") or {}).get("profiles", {}).get("1", {}).get("top_forms", [])[:2], ensure_ascii=False))

    job = asyncio_run(corpus_lexico(corpus, mineff=3))
    res = wait_job(job["job_id"])
    print("lexico status:", res["status"], "| spec rows:", len((res.get("result") or {}).get("specificities_forms", [])))

    # ---- matrix fixture (CHD needs enough rows/cols for its internal splits)
    import random

    random.seed(42)
    mpath = "tests/fixtures/mini_matrix.csv"
    sexes = ["h", "f"]
    ages = ["18-25", "26-35", "36-50"]
    themes = ["sport", "musique", "lecture", "cinema"]
    votes = ["A", "B"]
    with open(mpath, "w", encoding="utf8") as f:
        f.write("id;sex;age;theme;vote\n")
        for i in range(60):
            sex = sexes[i % 2]
            age = ages[i % 3]
            theme = themes[(i + i // 3) % 4]
            vote = "A" if (i % 2 == 0) == (theme in ("sport", "cinema")) else "B"
            f.write(f"m{i};{sex};{age};{theme};{vote}\n")

    pp("matrix_open", asyncio_run(matrix_open(mpath)))
    pp("matrix_frequencies", asyncio_run(matrix_frequencies(mpath, columns=[1, 2])))
    pp("matrix_chi2", asyncio_run(matrix_chi2(mpath, columns_a=[0], columns_b=[3])))

    job = asyncio_run(matrix_reinert(mpath, active_columns=[1, 2, 3], nbcl_p1=3))
    res = wait_job(job["job_id"])
    print("matrix_reinert status:", res["status"], "| classes:", (res.get("result") or {}).get("class_sizes"))

    job = asyncio_run(matrix_similitude(mpath, columns=[1, 2, 3]))
    res = wait_job(job["job_id"])
    print("matrix_similitude status:", res["status"], "| graphs:", (res.get("result") or {}).get("graphs"))

    pp("job_list", asyncio_run(job_list()))
    pp("analysis_list", {k: (len(v) if isinstance(v, list) else v) for k, v in asyncio_run(analysis_list()).items()})


if __name__ == "__main__":
    main()
