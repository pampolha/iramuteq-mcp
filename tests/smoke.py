"""Smoke test: bootstrap -> validate -> build corpus -> stats -> alceste (small)."""

import json
import sys

sys.path.insert(0, "src")

from iramuteq_mcp.bridge import bootstrap, patches, results, textops  # noqa: E402


def pp(label, obj):
    print(f"\n===== {label} =====")
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False)[:2000])


def main():
    ctx = bootstrap.init()
    print("bootstrap ok:", ctx.AppliPath, "|", ctx.UserConfigPath, "| R =", ctx.RPath)
    print("wx stub:", __import__("iramuteq_mcp.bridge.wxstub", fromlist=["x"]).is_installed())

    corpus_file = "tests/fixtures/mini_corpus.txt"

    pp("validate", textops.validate_corpus(corpus_file))

    corpus, csum = textops.build_corpus(corpus_file, lang="french", encoding="utf-8")
    pp("corpus", csum)

    pp("concordance", textops.concordance(corpus, "travail", limit=5))

    pp("stats", textops.run_stats(corpus_file, lang="french", encoding="utf-8"))

    print("\nalceste start...")
    res = textops.run_alceste(
        corpus_file,
        lang="french",
        encoding="utf-8",
        options={"classif_mode": 1, "nbcl_p1": 2, "max_actives": 200},
    )
    pp("alceste", res)


if __name__ == "__main__":
    main()
