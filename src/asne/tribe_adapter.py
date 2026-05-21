from __future__ import annotations

import hashlib
import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np

from .tts import TTSConfig, synthesize_text_to_audio


class AdapterUnavailableError(RuntimeError):
    """Raised when a requested neural encoding adapter cannot be used."""


class BaseNeuralEncodingAdapter(Protocol):
    def predict(self, stimulus_text: str, condition_id: str | None = None) -> dict:
        """Return a predicted cortical response for a stimulus and optional steering condition."""


@dataclass
class MockTribeAdapter:
    """Deterministic mock adapter for ASNE experiments before real TRIBE v2 integration."""

    response_size: int = 128

    def predict(self, stimulus_text: str, condition_id: str | None = None) -> dict:
        key = f"{stimulus_text}|{condition_id or 'baseline'}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)

        response = rng.normal(loc=0.0, scale=1.0, size=self.response_size)
        return {
            "model": "mock-tribe-adapter",
            "stimulus_text": stimulus_text,
            "condition_id": condition_id,
            "response": response,
        }


class TribeV2Adapter:
    """Placeholder for future TRIBE v2 integration."""

    VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
    TEXT_SUFFIXES = {".txt"}

    def __init__(
        self,
        model_name: str = "facebook/tribev2",
        cache_folder: str = "./cache",
        package_path: str | None = None,
        feature_device: str | None = None,
        tts_backend: str = "tribev2_gtts",
        tts_cache_dir: str = "./cache/asne_tts",
        tts_voice: str = "Samantha",
        tts_rate: int = 180,
        openai_tts_model: str = "gpt-4o-mini-tts",
        openai_tts_voice: str = "alloy",
        openai_tts_format: str = "mp3",
        higgs_model: str = "bosonai/higgs-audio-v2-generation-3B-base",
        higgs_tokenizer: str = "bosonai/higgs-audio-v2-tokenizer",
        higgs_device: str = "auto",
        higgs_max_new_tokens: int = 1024,
        higgs_temperature: float = 0.3,
        higgs_top_p: float = 0.95,
        higgs_top_k: int = 50,
        higgs_scene_description: str = "Audio is recorded from a quiet room.",
    ) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.package_path = str(Path(package_path).expanduser().resolve()) if package_path else None
        self.feature_device = feature_device
        self.tts_config = TTSConfig(
            backend=tts_backend,
            cache_dir=tts_cache_dir,
            voice=tts_voice,
            rate=tts_rate,
            openai_model=openai_tts_model,
            openai_voice=openai_tts_voice,
            openai_format=openai_tts_format,
            higgs_model=higgs_model,
            higgs_tokenizer=higgs_tokenizer,
            higgs_device=higgs_device,
            higgs_max_new_tokens=higgs_max_new_tokens,
            higgs_temperature=higgs_temperature,
            higgs_top_p=higgs_top_p,
            higgs_top_k=higgs_top_k,
            higgs_scene_description=higgs_scene_description,
        )
        self.model = None
        self._tribe_model_cls = self._load_tribe_model_class()

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("tribev2") is not None

    @contextmanager
    def _temporary_package_path(self):
        inserted = False
        if self.package_path and self.package_path not in sys.path:
            sys.path.insert(0, self.package_path)
            inserted = True
        try:
            yield
        finally:
            if inserted:
                try:
                    sys.path.remove(self.package_path)
                except ValueError:
                    pass

    @staticmethod
    def _is_namespace_shadow(module: ModuleType | None) -> bool:
        if module is None:
            return False
        return getattr(module, "__file__", None) is None and getattr(module, "__path__", None) is not None

    def _load_tribe_model_class(self):
        shadowed_module = sys.modules.get("tribev2")
        removed_shadow = None
        if self.package_path and self._is_namespace_shadow(shadowed_module):
            removed_shadow = sys.modules.pop("tribev2")

        if not self.package_path and not self.is_available():
            raise AdapterUnavailableError(
                "TRIBE v2 is not importable in this environment. "
                "Use '--adapter mock' for the current ASNE prototype, or install/configure "
                "TRIBE v2 separately before trying '--adapter tribev2'. ASNE does not "
                "install TRIBE v2, download model weights, or redistribute upstream artifacts."
            )

        try:
            with self._temporary_package_path():
                from tribev2 import TribeModel
        except Exception as exc:
            if removed_shadow is not None:
                sys.modules["tribev2"] = removed_shadow
            path_hint = (
                f" The supplied package_path was '{self.package_path}'."
                if self.package_path
                else ""
            )
            raise AdapterUnavailableError(
                "TRIBE v2 appears to be present, but 'from tribev2 import TribeModel' failed. "
                "This can happen when the local tribev2 checkout is not installed as a package "
                "or when its runtime dependencies are missing."
                f"{path_hint} Use '--adapter mock' for now, pass "
                "'--tribev2-package-path ./tribev2' for a local checkout, or install TRIBE v2 "
                "according to its upstream documentation and license."
            ) from exc

        return TribeModel

    def check_model_load(self, load_model: bool = False) -> dict:
        model_class_path = f"{self._tribe_model_cls.__module__}.{self._tribe_model_cls.__name__}"
        if not load_model:
            return {
                "available": True,
                "loaded": False,
                "model_name": self.model_name,
                "cache_folder": self.cache_folder,
                "model_class": model_class_path,
            }

        try:
            self.model = self._tribe_model_cls.from_pretrained(
                self.model_name,
                cache_folder=self.cache_folder,
            )
        except Exception as exc:
            raise AdapterUnavailableError(
                "TRIBE v2 model construction failed while calling "
                f"TribeModel.from_pretrained({self.model_name!r}, cache_folder={self.cache_folder!r}). "
                "This may require upstream model files, network access, compatible dependencies, "
                f"and sufficient local resources. Original error: {exc}"
            ) from exc

        return {
            "available": True,
            "loaded": True,
            "model_name": self.model_name,
            "cache_folder": self.cache_folder,
            "model_type": f"{type(self.model).__module__}.{type(self.model).__name__}",
            "model_class": model_class_path,
        }

    def force_feature_device(self, feature_device: str | None = None) -> list[dict]:
        """Best-effort device override for upstream TRIBE feature extractors."""

        requested_device = feature_device or self.feature_device
        if not requested_device or self.model is None:
            return []

        data = getattr(self.model, "data", None)
        changes: list[dict] = []
        for attr in ("text_feature", "audio_feature", "video_feature", "image_feature"):
            feature = getattr(data, attr, None)
            if feature is None:
                continue
            previous = getattr(feature, "device", None)
            if previous is not None:
                try:
                    setattr(feature, "device", requested_device)
                    changes.append(
                        {
                            "feature": attr,
                            "field": "device",
                            "previous": previous,
                            "new": getattr(feature, "device", None),
                        }
                    )
                except Exception as exc:
                    changes.append(
                        {
                            "feature": attr,
                            "field": "device",
                            "previous": previous,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            infra = getattr(feature, "infra", None)
            if infra is not None and requested_device == "cpu":
                previous_gpus = getattr(infra, "gpus_per_node", None)
                try:
                    setattr(infra, "gpus_per_node", 0)
                    changes.append(
                        {
                            "feature": attr,
                            "field": "infra.gpus_per_node",
                            "previous": previous_gpus,
                            "new": getattr(infra, "gpus_per_node", None),
                        }
                    )
                except Exception as exc:
                    changes.append(
                        {
                            "feature": attr,
                            "field": "infra.gpus_per_node",
                            "previous": previous_gpus,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        return changes

    @staticmethod
    def normalize_prediction(preds) -> tuple[np.ndarray, dict]:
        preds_array = np.asarray(preds, dtype=float)
        raw_shape = list(preds_array.shape)
        if preds_array.ndim == 0:
            response = preds_array.reshape(1)
            aggregation = "scalar_to_single_value"
        elif preds_array.ndim == 1:
            response = preds_array
            aggregation = "none_1d_response"
        elif preds_array.ndim == 2:
            response = np.mean(preds_array, axis=0)
            aggregation = "mean_over_time_axis_0"
        else:
            response = np.mean(preds_array.reshape((-1, preds_array.shape[-1])), axis=0)
            aggregation = "flatten_leading_axes_then_mean_over_axis_0"
        return response, {
            "raw_shape": raw_shape,
            "response_shape": list(response.shape),
            "aggregation": aggregation,
        }

    @classmethod
    def infer_input_mode(cls, stimulus_path: str | Path) -> str:
        suffix = Path(stimulus_path).suffix.lower()
        if suffix in cls.VIDEO_SUFFIXES:
            return "video"
        if suffix in cls.AUDIO_SUFFIXES:
            return "audio"
        if suffix in cls.TEXT_SUFFIXES:
            return "text"
        raise ValueError(
            "Unsupported TRIBE v2 smoke stimulus type. "
            f"Supported video suffixes: {sorted(cls.VIDEO_SUFFIXES)}; "
            f"supported audio suffixes: {sorted(cls.AUDIO_SUFFIXES)}; "
            f"supported text suffixes: {sorted(cls.TEXT_SUFFIXES)}. Got: {suffix or '<none>'}"
        )

    def predict(
        self,
        stimulus_path: str,
        condition_id: str | None = None,
        raw_prediction_path: str | Path | None = None,
    ) -> dict:
        path = Path(stimulus_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TRIBE v2 smoke stimulus file does not exist: {path}")
        input_mode = self.infer_input_mode(path)
        if self.model is None:
            self.check_model_load(load_model=True)
        feature_device_changes = self.force_feature_device()
        tts_metadata = None
        event_input_mode = input_mode

        if input_mode == "video":
            events = self.model.get_events_dataframe(video_path=str(path))
        elif input_mode == "audio":
            events = self.model.get_events_dataframe(audio_path=str(path))
        elif input_mode == "text":
            if self.tts_config.backend == "tribev2_gtts":
                events = self.model.get_events_dataframe(text_path=str(path))
            else:
                text = path.read_text(encoding="utf-8")
                if not text.strip():
                    raise ValueError(f"Text file is empty: {path}")
                tts_metadata = synthesize_text_to_audio(text, self.tts_config)
                events = self.model.get_events_dataframe(audio_path=tts_metadata["audio_path"])
                event_input_mode = "audio"
        else:
            raise ValueError(f"Unsupported input mode: {input_mode}")
        preds, segments = self.model.predict(events, verbose=False)
        preds_array = np.asarray(preds, dtype=float)
        saved_raw_prediction_path = None
        if raw_prediction_path is not None:
            raw_path = Path(raw_prediction_path)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(raw_path, preds_array)
            saved_raw_prediction_path = str(raw_path)
        response, normalization = self.normalize_prediction(preds)

        return {
            "adapter": "tribev2",
            "model": self.model_name,
            "condition_id": condition_id,
            "stimulus_path": str(path),
            "input_mode": input_mode,
            "response": response.astype(float).tolist(),
            "metadata": {
                **normalization,
                "segments_count": len(segments),
                "model_name": self.model_name,
                "cache_folder": self.cache_folder,
                "feature_device": self.feature_device,
                "feature_device_changes": feature_device_changes,
                "event_input_mode": event_input_mode,
                "tts_backend": self.tts_config.backend,
                "tts_metadata": tts_metadata,
                "raw_prediction_path": saved_raw_prediction_path,
                "warning": (
                    "TRIBE v2 smoke path only. Response is aggregated for ASNE "
                    "prototype compatibility; no steering condition has been applied. "
                    "Text inputs may trigger TRIBE or ASNE text-to-speech/audio preprocessing."
                ),
            },
        }


NeuralEncodingAdapter = BaseNeuralEncodingAdapter
