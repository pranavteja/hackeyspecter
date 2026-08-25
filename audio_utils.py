"""Efficient, safe audio helpers for recommendation playback and voice input."""
from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from rag_config import (
    LLM_PROVIDER,
    TTS_CACHE_SIZE,
    TTS_MAX_RETRIES,
    TTS_MODEL,
    TTS_TIMEOUT_SECONDS,
    TRANSCRIPTION_MODEL,
    get_client,
)
from story_display import story_summary_preview

logger = logging.getLogger(__name__)
TTS_VOICE = "alloy"
PLAYABLE_AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".ogg")
MAX_SUMMARY_SPEECH_CHARACTERS = 3_500
MAX_VOICE_INPUT_BYTES = 25 * 1024 * 1024


def is_playable_audio_url(url: str) -> bool:
    """Accept only direct HTTP(S) audio files, never ZIP/download pages."""
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.path.lower().endswith(PLAYABLE_AUDIO_SUFFIXES)
    )


def summary_for_speech(story: dict[str, Any], max_characters: int = MAX_SUMMARY_SPEECH_CHARACTERS) -> str:
    """Build a bounded narration script from a real stored story summary."""
    if not isinstance(story, dict):
        raise ValueError("The recommended story must be a record object.")
    if not isinstance(max_characters, int) or isinstance(max_characters, bool) or max_characters < 1:
        raise ValueError("max_characters must be a positive integer.")
    title = str(story.get("title", "This story")).strip() or "This story"
    summary = story_summary_preview(
        story.get("summary", ""),
        maximum_characters=max_characters,
    )
    if not summary:
        raise ValueError("The recommended story has no summary available for audio playback.")
    return f"{title}. {summary}"


@lru_cache(maxsize=TTS_CACHE_SIZE)
def _generate_tts_audio(script: str) -> bytes:
    """Cache repeated narration audio in-process to avoid duplicate billable TTS calls."""
    if LLM_PROVIDER != "openai":
        return _generate_local_tts_audio(script)
    if not script.strip():
        raise ValueError("Cannot generate speech from an empty script.")
    response = get_client().with_options(
        timeout=TTS_TIMEOUT_SECONDS,
        max_retries=TTS_MAX_RETRIES,
    ).audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=script,
        response_format="mp3",
    )
    audio_content = getattr(response, "content", None)
    if not isinstance(audio_content, (bytes, bytearray)) or not audio_content:
        raise ValueError("TTS model returned invalid audio data.")
    return bytes(audio_content)


def _generate_local_tts_audio(script: str) -> bytes:
    """Generate WAV audio with the operating system's offline speech engine."""
    if not script.strip():
        raise ValueError("Cannot generate speech from an empty script.")
    try:
        import pyttsx3
    except ImportError as exc:
        raise ValueError("Install offline text-to-speech support with: pip install pyttsx3") from exc

    output_path = ""
    audio_content = b""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = handle.name
        engine = pyttsx3.init()
        engine.setProperty("rate", int(os.getenv("LOCAL_TTS_RATE", "170")))
        engine.save_to_file(script, output_path)
        engine.runAndWait()
        with open(output_path, "rb") as audio_file:
            audio_content = audio_file.read()
    except Exception as exc:
        raise ValueError(f"Local text-to-speech failed: {exc}") from exc
    finally:
        if output_path:
            try:
                os.unlink(output_path)
            except OSError:
                pass
    if not audio_content:
        raise ValueError("Local text-to-speech returned empty audio.")
    return audio_content


def generate_summary_audio(story: dict[str, Any]) -> bytes:
    """Generate or reuse MP3 narration for a recommended story's stored summary."""
    script = summary_for_speech(story)
    logger.info("Generating or retrieving TTS summary audio for record %r", story.get("record_id"))
    return _generate_tts_audio(script)


def transcribe_voice_input(audio_bytes: bytes, filename: str, mime_type: str) -> str:
    """Convert a completed English microphone recording to a search query."""
    if not isinstance(audio_bytes, bytes):
        raise ValueError("Voice input must be binary audio data.")
    if not audio_bytes:
        raise ValueError("Record a voice message before transcribing it.")
    if len(audio_bytes) > MAX_VOICE_INPUT_BYTES:
        raise ValueError("Voice recording is too large. Record a shorter request.")
    if LLM_PROVIDER != "openai":
        return _transcribe_locally(audio_bytes, filename)
    logger.info("Transcribing microphone input: filename=%s, bytes=%d", filename, len(audio_bytes))
    transcription = get_client().audio.transcriptions.create(
        model=TRANSCRIPTION_MODEL,
        file=(filename, audio_bytes, mime_type),
        language="en",
        prompt="Transcribe the speaker's English story recommendation request accurately. Return English text only.",
    )
    text = transcription.text.strip()
    if not text:
        raise ValueError("The recording did not contain any transcribable speech.")
    logger.info("Microphone transcription completed successfully.")
    return text


def _transcribe_locally(audio_bytes: bytes, filename: str) -> str:
    """Transcribe locally with optional faster-whisper; never contacts a hosted API."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ValueError(
            "Install the free offline transcription engine first: "
            "pip install faster-whisper"
        ) from exc

    model_name = os.getenv("LOCAL_TRANSCRIPTION_MODEL", "small.en")
    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        suffix = os.path.splitext(filename or "recording.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio_bytes)
            temporary_path = handle.name
        try:
            segments, _ = model.transcribe(temporary_path, language="en", vad_filter=True)
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        finally:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
    except Exception as exc:
        raise ValueError(f"Local voice transcription failed: {exc}") from exc
    if not text:
        raise ValueError("The recording did not contain any transcribable speech.")
    logger.info("Local microphone transcription completed successfully using %s.", model_name)
    return text
