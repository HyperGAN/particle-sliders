"""Exercise the same HTTP requests the listening page's audio players make."""
import importlib.util
import threading
import urllib.error
import urllib.request
from pathlib import Path


def test_explicit_analysis_wav_links_are_served_without_exposing_other_files(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/listen_server.py"
    spec = importlib.util.spec_from_file_location("listen_server_test", script)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    root = tmp_path / "eval/listen"
    audio = root / "joint-v1/audio"
    audio.mkdir(parents=True)
    recordings = tmp_path / "analysis/run"
    recordings.mkdir(parents=True)
    wav = recordings / "audio.wav"
    payload = b"RIFF\x04\x00\x00\x00WAVE"
    wav.write_bytes(payload)
    (audio / "linked.wav").symlink_to(wav)
    (audio / "local.wav").write_bytes(payload)
    (recordings / "private.json").write_text('{"private":true}')
    (audio / "private.wav").symlink_to(recordings / "private.json")
    (audio / "external.wav").symlink_to(tmp_path / "external.wav")
    (tmp_path / "external.wav").write_bytes(payload)
    (audio / "directory").symlink_to(recordings, target_is_directory=True)
    (root / "joint-v1/index.html").write_text("listening page")
    server.ListenHandler.listen_root = root
    httpd = server.Server(("127.0.0.1", 0), server.ListenHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    try:
        for name in ("linked.wav", "local.wav"):
            url = f"{base}/joint-v1/audio/{name}?test=1"
            with urllib.request.urlopen(url) as response:
                assert response.status == 200
                assert response.url == url
                assert response.headers.get_content_type() in ("audio/x-wav", "audio/wav")
                assert response.read() == payload
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
                assert response.status == 200
                assert int(response.headers["Content-Length"]) == len(payload)
        with urllib.request.urlopen(f"{base}/joint-v1/") as response:
            assert response.read() == b"listening page"
        for name in ("private.wav", "external.wav", "directory/audio.wav", "missing.wav"):
            try:
                urllib.request.urlopen(f"{base}/joint-v1/audio/{name}")
            except urllib.error.HTTPError as exc:
                assert exc.code == 404
            else:
                raise AssertionError(f"Unexpectedly served {name}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
