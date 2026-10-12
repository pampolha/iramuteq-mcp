"""Result parsers that need neither R nor the IRaMuTeQ sources."""

from iramuteq_mcp.bridge import results


class _Obj:
    def __init__(self, pathout):
        self.parametres = {"pathout": pathout}


def test_parse_splitvar_picks_the_highest_numbered_directory(tmp_path):
    for n in (1, 2, 9, 10, 11, 12):
        d = tmp_path / f"A_matrix_{n}"
        d.mkdir()
        (d / "csvfile.csv").write_text("x")

    res = results.parse_splitvar(_Obj(str(tmp_path)), {}, ["A", "missing"])

    assert res["modalities"]["A"] == str(tmp_path / "A_matrix_12" / "csvfile.csv")
    assert res["modalities"]["missing"] is None


def test_parse_splitvar_without_output_directory():
    res = results.parse_splitvar(_Obj(None), {}, ["A"])
    assert res["modalities"] == {"A": None}


def test_trailing_number_orders_numerically():
    names = ["A_matrix_1", "A_matrix_10", "A_matrix_2", "A_matrix_11"]
    assert sorted(names, key=results._trailing_number) == [
        "A_matrix_1",
        "A_matrix_2",
        "A_matrix_10",
        "A_matrix_11",
    ]
