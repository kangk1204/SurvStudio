"""The CLI storage option reaches reload workers without changing analysis settings."""

from survival_toolkit import __main__ as command


def test_state_directory_is_forwarded_to_server(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(command, "_run_serve", lambda **kwargs: calls.append(kwargs) or 0)
    target = tmp_path / "private results"
    assert command.main(["serve", "--port", "8001", "--state-dir", str(target)]) == 0
    assert command.os.environ["SURVSTUDIO_STATE_DIR"] == str(target.absolute())
    assert calls == [{"host": "127.0.0.1", "port": 8001, "reload": False, "allowed_hosts": []}]


def test_inspect_does_not_start_or_create_saved_work(monkeypatch):
    monkeypatch.setattr(command, "_run_inspect", lambda path: 7)
    monkeypatch.setattr(command, "_run_serve", lambda **kwargs: (_ for _ in ()).throw(AssertionError("serve")))
    assert command.main(["inspect", "synthetic.csv"]) == 7
