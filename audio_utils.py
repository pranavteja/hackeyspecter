"""Efficient, safe audio helpers for recommendation playback and voice input."""
from __future__ import annotations

import base64
import binascii
import logging
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from rag_config import (
    TTS_CACHE_SIZE,
    TTS_MAX_RETRIES,
    TTS_MODEL,
    TTS_TIMEOUT_SECONDS,
    TRANSCRIPTION_MODEL,
    get_client,
)

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
    summary = str(story.get("summary", "")).strip()
    if not summary:
        raise ValueError("The recommended story has no summary available for audio playback.")
    return f"{title}. {summary[:max_characters]}"


@lru_cache(maxsize=TTS_CACHE_SIZE)
def _generate_tts_audio(script: str) -> bytes:
    """Cache repeated narration audio in-process to avoid duplicate billable TTS calls."""
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
    return response.content


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
