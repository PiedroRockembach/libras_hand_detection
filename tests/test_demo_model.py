import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_model.py"
spec = importlib.util.spec_from_file_location("demo_model", SCRIPT)
demo_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo_model)

RUN = "experiments/exp/final-x"


def make_run(root: Path) -> dict:
    run = root / RUN
    run.mkdir(parents=True)
    (run / "model.pt").write_bytes(b"\x00weights" * 100)
    (run / "meta.json").write_text('{"classes": ["A"]}')
    return {"run": RUN, "files": ["model.pt", "meta.json"],
            "release": {"repo": "o/r", "tag": "t", "asset": "m.tar.gz"},
            "sha256": None, "files_sha256": None}


def pack(tmp_path: Path, cfg: dict) -> tuple[Path, dict]:
    cfg_path = tmp_path / "v0.json"
    cfg_path.write_text(json.dumps(cfg))
    demo_model.cmd_pack(cfg_path, cfg, tmp_path / "src", tmp_path / "out")
    return tmp_path / "out" / "m.tar.gz", json.loads(cfg_path.read_text())


def test_pack_is_deterministic_and_fetch_restores_the_run(tmp_path):
    cfg = make_run(tmp_path / "src")
    asset, packed = pack(tmp_path, cfg)
    assert demo_model.build_archive(packed, tmp_path / "src")[0] == asset.read_bytes()
    assert packed["sha256"] == demo_model.sha256_file(asset)

    dst = tmp_path / "dst"
    dst.mkdir()
    demo_model.cmd_fetch(packed, dst, asset.as_uri())
    for f in cfg["files"]:
        assert (dst / RUN / f).read_bytes() == (tmp_path / "src" / RUN / f).read_bytes()
    assert [json.loads(line)["step"] for line in (dst / "PROVENANCE.jsonl").read_text().splitlines()] \
        == ["demo_model.fetch"]

    asset.unlink()  # already present: no download needed
    assert demo_model.cmd_fetch(packed, dst, asset.as_uri()) == 0


def test_fetch_refuses_an_archive_with_the_wrong_hash(tmp_path):
    asset, packed = pack(tmp_path, make_run(tmp_path / "src"))
    packed["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="mismatch"):
        demo_model.cmd_fetch(packed, tmp_path / "dst", asset.as_uri())
    assert not (tmp_path / "dst" / RUN).exists()
