"""Shared helpers for the test suite."""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from iramuteq_mcp import job_result, job_status

FIXTURES = Path(__file__).parent / "fixtures"


def run(coro):
    """Run one MCP tool coroutine to completion."""
    return asyncio.run(coro)


def wait_job(job_id: str, timeout: float = 900) -> dict:
    """Poll a job until it leaves queued/running, then return its full record."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = run(job_status(job_id))
        if st["status"] not in ("queued", "running"):
            return run(job_result(job_id))
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} still active after {timeout}s")


def write_matrix(path: Path) -> Path:
    """Deterministic 60-row individuals x variables table (see test fixtures)."""
    rng = random.Random(42)
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
            m1, m2 = rng.sample(words, 2)
            r1, r2 = rng.randint(1, 20), rng.randint(1, 20)
            f.write(f"m{i};{sex};{age};{theme};{vote};{m1};{m2};{r1};{r2}\n")
    return path


def write_large_corpus(path: Path, documents: int = 400) -> Path:
    """Synthetic ALCESTE corpus big enough for a classification run to last a while."""
    rng = random.Random(7)
    vocab = (
        "travail entreprise communication management formation equipe projet "
        "famille enfant ecole maison voyage musique lecture cinema sport jardin "
        "cuisine sante argent banque ville campagne television radio journal "
        "amitie confiance energie temps avenir regle probleme solution idee "
        "developpement recherche methode resultat science nature climat eau"
    ).split()
    with open(path, "w", encoding="utf8") as f:
        for i in range(documents):
            f.write(f"**** *sexe_{'h' if i % 2 else 'f'} *age_{20 + (i % 3) * 10}\n")
            for p in range(3):
                f.write(f"-*theme{p}\n")
                sentence = " ".join(rng.choice(vocab) for _ in range(60))
                f.write(sentence + ".\n")
    return path
