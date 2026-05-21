from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TTSGenerationError(RuntimeError):
    """Raised when ASNE cannot generate a cached audio stimulus."""


@dataclass(frozen=True)
class TTSConfig:
    backend: str = "tribev2_gtts"
    cache_dir: str = "./cache/asne_tts"
    voice: str = "Samantha"
    rate: int = 180
    openai_model: str = "gpt-4o-mini-tts"
    openai_voice: str = "alloy"
    openai_format: str = "mp3"
    higgs_model: str = "bosonai/higgs-audio-v2-generation-3B-base"
    higgs_tokenizer: str = "bosonai/higgs-audio-v2-tokenizer"
    higgs_device: str = "auto"
    higgs_max_new_tokens: int = 1024
    higgs_temperature: float = 0.3
    higgs_top_p: float = 0.95
    higgs_top_k: int = 50
    higgs_scene_description: str = "Audio is recorded from a quiet room."


_HIGGS_ENGINE_CACHE: dict[tuple[str, str, str], Any] = {}


def text_cache_key(text: str, config: TTSConfig) -> str:
    payload = {
        "text": text,
        "backend": config.backend,
        "voice": config.voice,
        "rate": config.rate,
        "openai_model": config.openai_model,
        "openai_voice": config.openai_voice,
        "openai_format": config.openai_format,
        "higgs_model": config.higgs_model,
        "higgs_tokenizer": config.higgs_tokenizer,
        "higgs_device": config.higgs_device,
        "higgs_max_new_tokens": config.higgs_max_new_tokens,
        "higgs_temperature": config.higgs_temperature,
        "higgs_top_p": config.higgs_top_p,
        "higgs_top_k": config.higgs_top_k,
        "higgs_scene_description": config.higgs_scene_description,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _metadata(audio_path: Path, text: str, config: TTSConfig) -> dict[str, Any]:
    return {
        "backend": config.backend,
        "audio_path": str(audio_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "voice": config.voice,
        "rate": config.rate,
        "openai_model": config.openai_model,
        "openai_voice": config.openai_voice,
        "openai_format": config.openai_format,
        "higgs_model": config.higgs_model,
        "higgs_tokenizer": config.higgs_tokenizer,
        "higgs_device": config.higgs_device,
        "higgs_max_new_tokens": config.higgs_max_new_tokens,
        "higgs_temperature": config.higgs_temperature,
        "higgs_top_p": config.higgs_top_p,
        "higgs_top_k": config.higgs_top_k,
        "higgs_scene_description": config.higgs_scene_description,
        "safety_note": (
            "This cached audio is a controlled stimulus rendering for ASNE predicted "
            "response-similarity experiments. It is not a measurement of an individual "
            "person's mental state."
        ),
    }


def synthesize_text_to_audio(text: str, config: TTSConfig) -> dict[str, Any]:
    """Generate or reuse cached audio for a text stimulus."""

    if config.backend == "tribev2_gtts":
        raise ValueError("tribev2_gtts is handled by TRIBE v2 TextToEvents, not ASNE TTS.")
    if config.backend not in {"macos_say", "openai", "higgs_audio"}:
        raise ValueError(f"Unsupported TTS backend: {config.backend}")

    cache_dir = Path(config.cache_dir) / config.backend
    cache_key = text_cache_key(text, config)
    suffix = ".wav" if config.backend in {"macos_say", "higgs_audio"} else f".{config.openai_format}"
    audio_path = cache_dir / f"{cache_key}{suffix}"
    metadata_path = cache_dir / f"{cache_key}.json"

    if audio_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["cache_hit"] = True
        return metadata

    cache_dir.mkdir(parents=True, exist_ok=True)
    if config.backend == "macos_say":
        _synthesize_macos_say(text, audio_path, config)
    elif config.backend == "openai":
        _synthesize_openai(text, audio_path, config)
    else:
        _synthesize_higgs_audio(text, audio_path, config)

    metadata = _metadata(audio_path, text, config)
    metadata["cache_hit"] = False
    _write_metadata(metadata_path, metadata)
    return metadata


def _synthesize_macos_say(text: str, audio_path: Path, config: TTSConfig) -> None:
    say_bin = shutil.which("say")
    afconvert_bin = shutil.which("afconvert")
    if say_bin is None:
        raise TTSGenerationError("macos_say backend requires the macOS 'say' command.")
    if afconvert_bin is None:
        raise TTSGenerationError("macos_say backend requires the macOS 'afconvert' command.")

    temp_aiff = audio_path.with_suffix(".aiff")
    try:
        subprocess.run(
            [
                say_bin,
                "--voice",
                config.voice,
                "--rate",
                str(config.rate),
                "--output-file",
                str(temp_aiff),
                text,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                afconvert_bin,
                str(temp_aiff),
                str(audio_path),
                "-f",
                "WAVE",
                "-d",
                "LEI16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise TTSGenerationError(
            f"macos_say synthesis failed with exit code {exc.returncode}: {stderr}"
        ) from exc
    finally:
        temp_aiff.unlink(missing_ok=True)


def _synthesize_openai(text: str, audio_path: Path, config: TTSConfig) -> None:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise TTSGenerationError(
            "openai TTS backend requires the optional 'openai' package. "
            "Install it with 'python -m pip install openai' or 'python -m pip install -e .[openai]'."
        ) from exc

    try:
        client = OpenAI()
        response = client.audio.speech.create(
            model=config.openai_model,
            voice=config.openai_voice,
            input=text,
            response_format=config.openai_format,
        )
        if hasattr(response, "write_to_file"):
            response.write_to_file(audio_path)
        elif hasattr(response, "read"):
            audio_path.write_bytes(response.read())
        else:
            audio_path.write_bytes(bytes(response))
    except Exception as exc:
        raise TTSGenerationError(
            "OpenAI TTS synthesis failed. Confirm OPENAI_API_KEY is set, network access is "
            f"available, and the selected model/voice are valid. Original error: {exc}"
        ) from exc


def _resolve_higgs_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_higgs_engine(config: TTSConfig):
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    except Exception as exc:
        raise TTSGenerationError(
            "higgs_audio backend requires the Boson AI higgs-audio package. "
            "Install it from https://github.com/boson-ai/higgs-audio, then retry. "
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    device = _resolve_higgs_device(config.higgs_device)
    cache_key = (config.higgs_model, config.higgs_tokenizer, device)
    if cache_key not in _HIGGS_ENGINE_CACHE:
        _HIGGS_ENGINE_CACHE[cache_key] = HiggsAudioServeEngine(
            config.higgs_model,
            config.higgs_tokenizer,
            device=device,
        )
    return _HIGGS_ENGINE_CACHE[cache_key], device


def _synthesize_higgs_audio(text: str, audio_path: Path, config: TTSConfig) -> None:
    try:
        import torch
        import torchaudio
        from boson_multimodal.data_types import ChatMLSample, Message
    except Exception as exc:
        raise TTSGenerationError(
            "higgs_audio backend requires torch, torchaudio, and the Boson AI higgs-audio package. "
            "Install the upstream package and confirm it can import before running ASNE. "
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        engine, _device = _get_higgs_engine(config)
        system_prompt = (
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\n"
            f"{config.higgs_scene_description}\n"
            "<|scene_desc_end|>"
        )
        output = engine.generate(
            chat_ml_sample=ChatMLSample(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=text),
                ]
            ),
            max_new_tokens=config.higgs_max_new_tokens,
            temperature=config.higgs_temperature,
            top_p=config.higgs_top_p,
            top_k=config.higgs_top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        torchaudio.save(
            str(audio_path),
            torch.from_numpy(output.audio)[None, :],
            output.sampling_rate,
        )
    except Exception as exc:
        raise TTSGenerationError(
            "Higgs Audio synthesis failed. Confirm model files are available, the selected "
            f"device is supported, and local memory is sufficient. Original error: {exc}"
        ) from exc
