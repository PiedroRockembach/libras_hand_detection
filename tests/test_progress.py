import json

from libras.progress import Progress


def test_progress_file_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_PROGRESS_DIR", str(tmp_path))
    with Progress("job", "Um trabalho", total=3, unit="itens") as p:
        p.advance()
        data = json.loads((tmp_path / "job.json").read_text())
        assert data["status"] == "running" and data["total"] == 3 and data["unit"] == "itens"
        p.advance(2)
    data = json.loads((tmp_path / "job.json").read_text())
    assert data["status"] == "done" and data["done"] == 3  # the final write is never throttled
    assert not list(tmp_path.glob("*.tmp"))


def test_progress_marks_error_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_PROGRESS_DIR", str(tmp_path / "file-not-dir"))
    (tmp_path / "file-not-dir").write_text("x")  # unwritable location: reporting must not raise
    Progress("job", "x").finish(ok=False)
