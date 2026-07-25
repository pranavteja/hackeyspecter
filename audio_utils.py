"""Audio helpers for recommendation playback."""
from __future__ import annotations

import logging
from typing import Any

from rag_config import get_client

logger = logging.getLogger(__name__)
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
# Use OpenAI's higher-accuracy transcription model for microphone requests.
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
PLAYABLE_AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".ogg")


def is_playable_audio_url(url: str) -> bool:
    """Return True only for direct audio files, not archive/download pages."""
    return url.lower().split("?", 1)[0].endswith(PLAYABLE_AUDIO_SUFFIXES)


def summary_for_speech(story: dict[str, Any], max_characters: int = 3_500) -> str:
    """Build a concise TTS script from a real stored story summary."""
    title = str(story.get("title", "This story"))
    summary = str(story.get("summary", "")).strip()
    if not summary:
        raise ValueError("The recommended story has no summary available for audio playback.")
    return f"{title}. {summary[:max_characters]}"


def generate_summary_audio(story: dict[str, Any]) -> bytes:
    """Generate MP3 narration for the stored summary using OpenAI Text-to-Speech."""
    script = summary_for_speech(story)
    logger.info("Generating TTS summary audio for %r", story.get("title"))
    response = get_client().audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=script,
        response_format="mp3",
    )
    return response.read()


def transcribe_voice_input(audio_bytes: bytes, filename: str, mime_type: str) -> str:
    """Convert microphone audio to a search query with OpenAI transcription."""
    if not audio_bytes:
        raise ValueError("Record a voice message before transcribing it.")
    logger.info("Transcribing microphone input: filename=%s, bytes=%d", filename, len(audio_bytes))
    transcription = get_client().audio.transcriptions.create(
        model=TRANSCRIPTION_MODEL,
        file=(filename, audio_bytes, mime_type),
        language="en",
        prompt="Transcribe the speaker's English request accurately. Return English text only.",
    )
    text = transcription.text.strip()
    if not text:
        raise ValueError("The recording did not contain any transcribable speech.")
    logger.info("Microphone transcription completed: %r", text)
    return text
