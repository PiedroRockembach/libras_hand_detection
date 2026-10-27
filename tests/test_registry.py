from pathlib import Path

import pytest

from libras.registry.schema import TABLES
from libras.registry.store import (
    CURATED_DIR,
    load_registry,
    read_table,
    snapshot,
    split_group,
    validate,
    write_table,
)


def make_registry():
    return {
        "sources": [
            {"source_id": "a", "name": "A", "url": "u", "license": "CC-BY-4.0",
             "redistribution": "yes", "commercial_use": "yes"},
        ],
        "glosses": [{"gloss_id": "G1", "gloss": "CASA", "unit": "sign"}],
        "label_map": [{"source_id": "a", "source_label": "casa", "gloss_id": "G1"}],
        "signers": [
            {"signer_uid": "a:1", "source_id": "a", "source_signer_id": "1"},
            {"signer_uid": "a:2", "source_id": "a", "source_signer_id": "2",
             "same_person_as": "a:1"},
        ],
        "samples": [
            {"sample_id": "a:1:casa:1", "source_id": "a", "gloss_id": "G1", "signer_uid": "a:1",
             "path": "raw/a/x.mp4", "media": "video", "fps": "30", "license": "CC-BY-4.0",
             "consent_scope": "public_license"},
        ],
    }


def test_valid_registry_has_no_errors():
    assert validate(make_registry()) == []


def test_detects_missing_bad_choice_kind_fk_and_duplicates():
    reg = make_registry()
    reg["sources"][0]["redistribution"] = "maybe"
    reg["samples"][0]["fps"] = "thirty"
    reg["samples"][0]["gloss_id"] = "G404"
    reg["samples"].append(dict(reg["samples"][0]))
    reg["glosses"].append({"gloss_id": "G2"})
    errors = "\n".join(validate(reg))
    for expected in ("redistribution='maybe'", "fps='thirty'", "gloss_id='G404'",
                     "duplicate key", "missing gloss"):
        assert expected in errors


def test_split_group_follows_same_person_links():
    groups = split_group(make_registry())
    assert groups == {"a:1": "a:1", "a:2": "a:1"}


def test_csv_roundtrip_and_unknown_columns(tmp_path: Path):
    table = TABLES["glosses"]
    write_table(tmp_path / "g.csv", table, [{"gloss_id": "G1", "gloss": "CASA", "unit": "sign"}])
    assert read_table(tmp_path / "g.csv", table)[0]["gloss"] == "CASA"
    (tmp_path / "bad.csv").write_text("gloss_id,gloss,oops\nG1,CASA,x\n")
    with pytest.raises(ValueError, match="unknown columns"):
        read_table(tmp_path / "bad.csv", table)


def test_snapshot_is_immutable(tmp_path: Path):
    digest = snapshot(make_registry(), tmp_path, "v0.1")
    assert len(digest) == 64
    assert (tmp_path / "registry" / "v0.1" / "SHA256SUMS").exists()
    with pytest.raises(FileExistsError):
        snapshot(make_registry(), tmp_path, "v0.1")


def test_curated_tables_in_repo_are_valid():
    reg = load_registry(None, CURATED_DIR)
    assert reg["sources"], "metadata/registry/sources.csv should list the downloaded sources"
    assert validate(reg) == []
