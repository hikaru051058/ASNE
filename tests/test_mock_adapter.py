import sys

import numpy as np
import pytest

from asne.cli import check_adapter
from asne.tribe_adapter import AdapterUnavailableError, MockTribeAdapter, TribeV2Adapter


def test_mock_adapter_is_deterministic_for_same_inputs():
    adapter = MockTribeAdapter(response_size=16)

    first = adapter.predict("A person walks into a dark alley.", condition_id="threat")
    second = adapter.predict("A person walks into a dark alley.", condition_id="threat")

    np.testing.assert_allclose(first["response"], second["response"])
    assert first["response"].shape == (16,)


def test_mock_adapter_changes_with_condition_id():
    adapter = MockTribeAdapter(response_size=16)

    neutral = adapter.predict("A person walks into a dark alley.", condition_id="neutral")
    threat = adapter.predict("A person walks into a dark alley.", condition_id="threat")

    assert not np.allclose(neutral["response"], threat["response"])


def test_tribev2_adapter_unavailable_message_when_import_fails(monkeypatch):
    monkeypatch.setattr(TribeV2Adapter, "is_available", staticmethod(lambda: False))

    with pytest.raises(AdapterUnavailableError, match="TRIBE v2 is not importable"):
        TribeV2Adapter()


def test_tribev2_adapter_import_entrypoint_failure_is_clear(monkeypatch):
    monkeypatch.setattr(TribeV2Adapter, "is_available", staticmethod(lambda: True))

    with pytest.raises(AdapterUnavailableError, match="TribeModel"):
        TribeV2Adapter()


def test_tribev2_adapter_accepts_package_path_without_permanent_sys_path(monkeypatch, tmp_path):
    package_root = tmp_path / "local_tribev2"
    package_dir = package_root / "tribev2"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "class TribeModel:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "tribev2", raising=False)

    adapter = TribeV2Adapter(package_path=str(package_root))

    assert adapter._tribe_model_cls.__name__ == "TribeModel"
    assert str(package_root.resolve()) not in sys.path
    monkeypatch.delitem(sys.modules, "tribev2", raising=False)


def test_tribev2_check_model_load_false_returns_class_metadata(monkeypatch, tmp_path):
    package_root = tmp_path / "local_tribev2"
    package_dir = package_root / "tribev2"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "class TribeModel:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "tribev2", raising=False)

    adapter = TribeV2Adapter(package_path=str(package_root))
    result = adapter.check_model_load(load_model=False)

    assert result["available"] is True
    assert result["loaded"] is False
    assert result["model_class"] == "tribev2.TribeModel"
    monkeypatch.delitem(sys.modules, "tribev2", raising=False)


def test_tribev2_check_model_load_true_uses_from_pretrained(monkeypatch, tmp_path):
    package_root = tmp_path / "local_tribev2"
    package_dir = package_root / "tribev2"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "class LoadedModel:\n"
        "    pass\n"
        "\n"
        "class TribeModel:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, model_name, cache_folder=None):\n"
        "        return LoadedModel()\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "tribev2", raising=False)

    adapter = TribeV2Adapter(
        model_name="local-model",
        cache_folder="./cache-test",
        package_path=str(package_root),
    )
    result = adapter.check_model_load(load_model=True)

    assert result["available"] is True
    assert result["loaded"] is True
    assert result["model_name"] == "local-model"
    assert result["cache_folder"] == "./cache-test"
    assert result["model_type"] == "tribev2.LoadedModel"
    monkeypatch.delitem(sys.modules, "tribev2", raising=False)


def test_mock_check_adapter_path():
    result = check_adapter("mock")

    assert "Adapter available: mock" in result
    assert "response_size=128" in result


def test_tribev2_check_adapter_import_path(monkeypatch, tmp_path):
    package_root = tmp_path / "local_tribev2"
    package_dir = package_root / "tribev2"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "class TribeModel:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "tribev2", raising=False)

    result = check_adapter("tribev2", tribev2_package_path=str(package_root))

    assert "Adapter import available: tribev2" in result
    assert "TribeModel=tribev2.TribeModel" in result
    monkeypatch.delitem(sys.modules, "tribev2", raising=False)


def test_tribev2_normalize_prediction_means_over_time_axis():
    preds = np.array([[1.0, 3.0, 5.0], [3.0, 5.0, 7.0]])

    response, metadata = TribeV2Adapter.normalize_prediction(preds)

    np.testing.assert_allclose(response, np.array([2.0, 4.0, 6.0]))
    assert metadata["raw_shape"] == [2, 3]
    assert metadata["response_shape"] == [3]
    assert metadata["aggregation"] == "mean_over_time_axis_0"


def test_tribev2_normalize_prediction_keeps_1d_response():
    preds = np.array([1.0, 2.0, 3.0])

    response, metadata = TribeV2Adapter.normalize_prediction(preds)

    np.testing.assert_allclose(response, preds)
    assert metadata["raw_shape"] == [3]
    assert metadata["aggregation"] == "none_1d_response"


def test_tribev2_infer_input_mode_supports_video_audio_and_text():
    assert TribeV2Adapter.infer_input_mode("sample.mp4") == "video"
    assert TribeV2Adapter.infer_input_mode("sample.webm") == "video"
    assert TribeV2Adapter.infer_input_mode("sample.wav") == "audio"
    assert TribeV2Adapter.infer_input_mode("sample.mp3") == "audio"
    assert TribeV2Adapter.infer_input_mode("sample.txt") == "text"

    with pytest.raises(ValueError, match="Unsupported TRIBE v2 smoke stimulus type"):
        TribeV2Adapter.infer_input_mode("sample.pdf")


def test_tribev2_predict_routes_text_path_without_real_inference(monkeypatch, tmp_path):
    text_path = tmp_path / "stimulus.txt"
    text_path.write_text("A short local text stimulus.", encoding="utf-8")

    class FakeModel:
        def __init__(self):
            self.called_with = None

        def get_events_dataframe(self, text_path=None, video_path=None):
            self.called_with = {"text_path": text_path, "video_path": video_path}
            return {"events": "fake"}

        def predict(self, events, verbose=True):
            assert events == {"events": "fake"}
            assert verbose is False
            return np.array([[1.0, 2.0], [3.0, 4.0]]), ["s1", "s2"]

    monkeypatch.setattr(TribeV2Adapter, "_load_tribe_model_class", lambda self: object)
    adapter = TribeV2Adapter()
    adapter.model = FakeModel()

    result = adapter.predict(str(text_path), condition_id=None)

    assert adapter.model.called_with == {"text_path": str(text_path), "video_path": None}
    assert result["adapter"] == "tribev2"
    assert result["input_mode"] == "text"
    assert result["metadata"]["raw_shape"] == [2, 2]
    assert result["metadata"]["segments_count"] == 2
    assert result["response"] == [2.0, 3.0]


def test_tribev2_predict_routes_text_via_asne_tts_audio(monkeypatch, tmp_path):
    text_path = tmp_path / "stimulus.txt"
    text_path.write_text("A short local text stimulus.", encoding="utf-8")
    audio_path = tmp_path / "cached.wav"
    audio_path.write_bytes(b"RIFFfake")

    class FakeModel:
        def __init__(self):
            self.called_with = None

        def get_events_dataframe(self, text_path=None, audio_path=None, video_path=None):
            self.called_with = {
                "text_path": text_path,
                "audio_path": audio_path,
                "video_path": video_path,
            }
            return {"events": "fake"}

        def predict(self, events, verbose=True):
            assert events == {"events": "fake"}
            assert verbose is False
            return np.array([[1.0, 2.0], [3.0, 4.0]]), ["s1", "s2"]

    monkeypatch.setattr(TribeV2Adapter, "_load_tribe_model_class", lambda self: object)
    monkeypatch.setattr(
        "asne.tribe_adapter.synthesize_text_to_audio",
        lambda text, config: {
            "backend": config.backend,
            "audio_path": str(audio_path),
            "cache_hit": False,
        },
    )
    adapter = TribeV2Adapter(tts_backend="macos_say")
    adapter.model = FakeModel()

    result = adapter.predict(str(text_path), condition_id=None)

    assert adapter.model.called_with == {
        "text_path": None,
        "audio_path": str(audio_path),
        "video_path": None,
    }
    assert result["input_mode"] == "text"
    assert result["metadata"]["event_input_mode"] == "audio"
    assert result["metadata"]["tts_backend"] == "macos_say"
    assert result["metadata"]["tts_metadata"]["audio_path"] == str(audio_path)
