from pathlib import Path

from asne.tts import TTSConfig, synthesize_text_to_audio, text_cache_key


def test_text_cache_key_changes_by_backend() -> None:
    text = "A person checks a sign beside the door."

    first = text_cache_key(text, TTSConfig(backend="macos_say"))
    second = text_cache_key(text, TTSConfig(backend="openai"))

    assert first != second


def test_text_cache_key_changes_by_higgs_settings() -> None:
    text = "A person checks a sign beside the door."

    first = text_cache_key(text, TTSConfig(backend="higgs_audio", higgs_device="mps"))
    second = text_cache_key(text, TTSConfig(backend="higgs_audio", higgs_device="cpu"))

    assert first != second


def test_macos_say_writes_cached_metadata(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_which(name: str) -> str:
        return f"/usr/bin/{name}"

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        if "afconvert" in cmd[0]:
            Path(cmd[2]).write_bytes(b"RIFFfake")
        else:
            Path(cmd[cmd.index("--output-file") + 1]).write_bytes(b"FORMfake")

    monkeypatch.setattr("asne.tts.shutil.which", fake_which)
    monkeypatch.setattr("asne.tts.subprocess.run", fake_run)

    config = TTSConfig(backend="macos_say", cache_dir=str(tmp_path), voice="Samantha", rate=180)
    first = synthesize_text_to_audio("A short controlled text.", config)
    second = synthesize_text_to_audio("A short controlled text.", config)

    assert first["backend"] == "macos_say"
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert Path(first["audio_path"]).exists()
    assert len(calls) == 2
