import csv
from pathlib import Path

from libras.registry.annotation import import_kit
from libras.registry.schema import TABLES
from libras.registry.store import load_registry, read_table, validate, write_table

REVIEW = ["gloss_id", "glosa_atual", "tipo_atual", "rotulos_originais", "video_1", "video_2",
          "aviso", "glosa_corrigida", "tipo_corrigido", "variante_regional", "observacoes"]
PHRASES = ["gloss_id", "frase_portugues", "n_videos", "video_1", "video_2", "video_3", "aviso",
           "sequencia_glosas_libras", "observacoes"]


def setup(tmp_path: Path, review: list[dict], phrases: list[dict]) -> tuple[Path, Path]:
    curated, kit = tmp_path / "curated", tmp_path / "kit-v9"
    curated.mkdir()
    kit.mkdir()
    write_table(curated / "glosses.csv", TABLES["glosses"], [
        {"gloss_id": "G00001", "gloss": "MACA", "unit": "sign", "notes": "auto; review"},
        {"gloss_id": "G00002", "gloss": "POSTE", "unit": "sign"},
        {"gloss_id": "G00003", "gloss": "CAIR", "unit": "sign"},
        {"gloss_id": "G00004", "gloss": "UM POSTE CAIU", "unit": "phrase"},
        {"gloss_id": "G00005", "gloss": "NÃO SEI", "unit": "phrase"},
    ])
    write_table(curated / "phrase_glosses.csv", TABLES["phrase_glosses"], [])
    for name, header, rows in (("revisao_glosas.csv", REVIEW, review),
                               ("anotacao_frases.csv", PHRASES, phrases)):
        # Saved the way LibreOffice may save it: BOM and semicolons.
        with (kit / name).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=header, delimiter=";")
            w.writeheader()
            w.writerows(rows)
    return curated, kit


def row(gid: str, current: str, **kw) -> dict:
    return {"gloss_id": gid, "glosa_atual": current, **kw}


def test_import_applies_review_and_sequences(tmp_path: Path):
    curated, kit = setup(
        tmp_path,
        review=[row("G00001", "MACA", glosa_corrigida="maçã"),
                row("G00005", "NÃO SEI", tipo_corrigido="sinal"),
                row("G00002", "POSTE", observacoes="ok"),
                row("G00003", "CAIR")],  # left blank: not reviewed
        phrases=[{"gloss_id": "G00004", "sequencia_glosas_libras": "poste  CAIR+ p-i-x ?"}],
    )
    rep = import_kit(kit, "AB", curated, apply=True)
    assert rep.errors == []
    glosses = {g["gloss_id"]: g for g in read_table(curated / "glosses.csv", TABLES["glosses"])}
    assert glosses["G00001"]["gloss"] == "MAÇÃ"
    assert "review" not in glosses["G00001"]["notes"]
    assert glosses["G00005"]["unit"] == "sign"
    assert "revisado" in glosses["G00002"]["notes"] and "ok" not in glosses["G00002"]["notes"]
    assert "revisado" not in glosses["G00003"]["notes"]
    assert glosses["G00006"]["gloss"] == "P-I-X"  # new sign created from the phrase
    (p,) = read_table(curated / "phrase_glosses.csv", TABLES["phrase_glosses"])
    assert p["sequence"] == "G00002 G00003 G00006"
    assert p["complete"] == "no" and p["sequence_text"] == "POSTE CAIR+ P-I-X ?"
    assert validate(load_registry(None, curated)) == []


def test_dry_run_writes_nothing_and_errors_block(tmp_path: Path):
    curated, kit = setup(
        tmp_path,
        review=[row("G00001", "OUTRA"), row("G00002", "POSTE", glosa_corrigida="CAIR")],
        phrases=[{"gloss_id": "G00004", "sequencia_glosas_libras": "POSTE caiu!"}],
    )
    before = (curated / "glosses.csv").read_text()
    rep = import_kit(kit, "AB", curated, apply=True)
    joined = "\n".join(rep.errors)
    assert "changed since the kit" in joined
    assert "would duplicate" in joined
    assert "outside the conventions" in joined
    assert (curated / "glosses.csv").read_text() == before
