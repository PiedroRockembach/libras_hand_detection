from pathlib import Path

from libras.registry.adapters import eqt_uece, malta, minds
from libras.registry.adapters.base import gloss_text, match_key
from libras.registry.build import build, flag_duplicates, propose_labels
from libras.registry.store import load_registry, validate


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def fake_data_root(root: Path) -> Path:
    raw = root / "raw"
    touch(raw / "minds_libras_rgb/extracted/Sinalizador03/Canon/15MacaSinalizador03-2.mp4")
    touch(raw / "minds_libras_rgbd/extracted/Sinalizador03/15Maca/3-15Maca_2RGB.mp4")
    eqt = raw / "libras_eqt_uece_v4/extracted/Libras-EQT-UECE (RGB)"
    touch(eqt / "99_Oscilar./INFORMANTE 5 /IMG_1.mp4")
    touch(eqt / "114_Oscilar./INFORMANTE 1/IMG_2.mp4")
    touch(eqt / "18_Quero alterar a forma de pagamento\npara Pix./INFORMANTE 2/IMG_3.mp4")
    mal = raw / "malta_libras/malta_libras_complete"
    touch(mal / "ufsc/B%C3%81SICO.pt")
    touch(mal / "acessibilidade_3/macaSm_Prog001.pt")
    touch(mal / "youtube/Libras - Parque de Divers├úo (1080p_30fps_H264-128kbit_AAC).pt")
    return root


def test_gloss_normalization():
    assert gloss_text("Quero alterar a forma de pagamento\npara Pix.") == (
        "QUERO ALTERAR A FORMA DE PAGAMENTO PARA PIX")
    assert match_key("Maçã") == match_key("Maca") == "MACA"


def test_minds_links_kinect_signers_to_canon(tmp_path: Path):
    res = minds.scan(fake_data_root(tmp_path))
    ids = {s["sample_id"]: s for s in res.samples}
    assert set(ids) == {"minds:03:maca:2", "minds_kinect:03:maca:2"}
    kinect = next(s for s in res.signers if s["signer_uid"] == "minds_kinect:03")
    assert kinect["same_person_as"] == "minds:03"


def test_eqt_normalizes_informant_and_labels(tmp_path: Path):
    res = eqt_uece.scan(fake_data_root(tmp_path))
    assert {s["signer_uid"] for s in res.signers} == {"eqt_uece:1", "eqt_uece:2", "eqt_uece:5"}
    labels = {s["source_label"] for s in res.samples}
    assert "Quero alterar a forma de pagamento para Pix." in labels


def test_malta_label_cleanup(tmp_path: Path):
    labels = {s["source_label"] for s in malta.scan(fake_data_root(tmp_path)).samples}
    assert labels == {"BÁSICO", "maca", "Parque de Diversão"}


def test_flag_duplicates_marks_label_conflicts():
    samples = [
        {"sample_id": "a", "gloss_id": "G1", "video_sha256": "h1"},
        {"sample_id": "b", "gloss_id": "G1", "video_sha256": "h1"},
        {"sample_id": "c", "gloss_id": "G2", "video_sha256": "h2", "quality_flags": "x"},
        {"sample_id": "d", "gloss_id": "G3", "video_sha256": "h2"},
        {"sample_id": "e", "gloss_id": "G1", "video_sha256": "h3"},
    ]
    flag_duplicates(samples)
    flags = {s["sample_id"]: s.get("quality_flags", "") for s in samples}
    assert flags == {"a": "duplicate_video", "b": "duplicate_video",
                     "c": "duplicate_video;label_conflict;x",
                     "d": "duplicate_video;label_conflict", "e": ""}


def test_propose_and_build_produce_valid_registry(tmp_path: Path):
    root = fake_data_root(tmp_path / "data")
    curated = tmp_path / "curated"
    (curated).mkdir()
    (curated / "sources.csv").write_text(
        "source_id,name,url,license,redistribution,commercial_use\n"
        "minds,M,u,CC-BY-4.0,yes,yes\nminds_kinect,MK,u,CC-BY-4.0,yes,yes\n"
        "eqt_uece,E,u,CC-BY-NC-4.0,yes,no\nmalta,MA,u,MIT,unknown,unknown\n")
    names = ["minds", "eqt_uece", "malta"]
    added = propose_labels([f(root) for f in (minds.scan, eqt_uece.scan, malta.scan)], curated)
    # Maca (MINDS) and maca (MALTA) share one gloss; the two Oscilar classes share one.
    assert added == {"glosses": 5, "label_map": 7}

    report = build(root, names, curated_dir=curated, probe=False)
    assert report["unmapped_samples"] == 0
    reg = load_registry(root, curated)
    assert validate(reg) == []
    assert len(reg["samples"]) == 8
