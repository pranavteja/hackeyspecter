"""Fast, offline regression tests for RAG correctness and safety invariants."""
from __future__ import annotations

import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from audio_utils import _generate_tts_audio, is_playable_audio_url, summary_for_speech
from mood_engine.check_intent import check_intent
from preprocess_stories import _embed_payloads, _extract_features_with_retry
from recommend_and_pitch import generate_final_recommendation
from search_engine import (
    StoryDatabase,
    _embed_normalized_query,
    _get_normalized_query,
    canonicalize_query,
    load_database,
    vector_search,
)


class VectorSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = StoryDatabase(
            records=(
                {"record_id": "a", "title": "Closest"},
                {"record_id": "b", "title": "Second"},
                {"record_id": "c", "title": "Opposite"},
            ),
            embeddings=np.asarray([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]], dtype=np.float32),
            embedding_model="test-embedding",
            embedding_dimensions=2,
            build_fingerprint="test-store",
        )

    @patch("search_engine._embed_normalized_query", return_value=(1.0, 0.0))
    def test_returns_descending_top_k(self, mock_embed: object) -> None:
        results = vector_search("calm story", self.database, top_k=2)
        self.assertEqual([result["record_id"] for result in results], ["a", "b"])
        self.assertGreater(results[0]["vector_similarity"], results[1]["vector_similarity"])

    def test_rejects_empty_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "Enter a story"):
            vector_search("   ", self.database)

    def test_rejects_non_integer_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be an integer"):
            vector_search("calm story", self.database, top_k=1.5)  # type: ignore[arg-type]

    def test_rejects_inconsistent_in_memory_database(self) -> None:
        inconsistent = StoryDatabase(
            records=self.database.records,
            embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            embedding_model="test-embedding",
            embedding_dimensions=2,
            build_fingerprint="bad-store",
        )
        with self.assertRaisesRegex(ValueError, "inconsistent record"):
            vector_search("calm story", inconsistent)

    def test_canonicalize_query_collapses_only_harmless_whitespace(self) -> None:
        self.assertEqual(
            canonicalize_query("  A\t rainy\nSunday   after heartbreak  "),
            "A rainy Sunday after heartbreak",
        )

    @patch("search_engine.get_client")
    def test_query_embedding_cache_keeps_compact_immutable_float32_vectors(self, mock_client: object) -> None:
        _embed_normalized_query.cache_clear()
        embedding_client = mock_client.return_value.with_options.return_value
        embedding_client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[3.0, 4.0])]
        )
        first = _embed_normalized_query("calm", "test-embedding", 2)
        second = _embed_normalized_query("calm", "test-embedding", 2)

        self.assertIs(first, second)
        self.assertEqual(first.dtype, np.float32)
        self.assertFalse(first.flags.writeable)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=6)
        self.assertEqual(embedding_client.embeddings.create.call_count, 1)

    @patch("search_engine.get_client")
    def test_concurrent_identical_queries_share_one_embedding_request(self, mock_client: object) -> None:
        _embed_normalized_query.cache_clear()
        call_count = 0
        count_lock = threading.Lock()

        def create_embedding(**_: object) -> SimpleNamespace:
            nonlocal call_count
            with count_lock:
                call_count += 1
            time.sleep(0.05)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])

        mock_client.return_value.with_options.return_value.embeddings.create.side_effect = create_embedding
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(_get_normalized_query, "same query", "test-embedding", 2)
            second_future = executor.submit(_get_normalized_query, "same query", "test-embedding", 2)
            first, second = first_future.result(), second_future.result()

        self.assertEqual(call_count, 1)
        self.assertTrue(np.array_equal(first[0], second[0]))

    def test_load_real_legacy_store_adds_stable_ids_and_normalizes_vectors(self) -> None:
        path = Path(__file__).resolve().parents[1] / "stories_vector_store.npz"
        database = load_database(str(path))
        record_ids = [record["record_id"] for record in database.records]
        self.assertTrue(all(record_ids))
        self.assertEqual(len(record_ids), len(set(record_ids)))
        self.assertAlmostEqual(float(database.embeddings[0] @ database.embeddings[0]), 1.0, places=6)


class RecommendationTests(unittest.TestCase):
    @patch("recommend_and_pitch.get_client")
    def test_derives_title_and_audio_url_from_selected_record(self, mock_client: object) -> None:
        first = {
            "record_id": "first", "title": "Duplicate Title", "summary": "First summary.",
            "audio_zip_url": "https://example.test/first.zip", "vector_similarity": 0.9,
        }
        second = {
            "record_id": "second", "title": "Duplicate Title", "summary": "Second summary.",
            "audio_zip_url": "https://example.test/second.zip", "vector_similarity": 0.8,
        }
        response = SimpleNamespace(output_text=json.dumps({
            "recommended_record_id": "second",
            "pitch_script": "It matches your mood. Its arc gives you hope.",
            "emotional_match_reasons": ["Hopeful emotional arc"],
        }))
        mock_client.return_value.responses.create.return_value = response

        result = generate_final_recommendation("I need hope", [first, second])
        self.assertEqual(result["recommended_record_id"], "second")
        self.assertEqual(result["audio_url"], "https://example.test/second.zip")
        self.assertEqual(result["recommended_book_title"], "Duplicate Title")

    @patch("recommend_and_pitch.get_client")
    def test_rejects_model_choice_outside_candidates(self, mock_client: object) -> None:
        candidate = {"record_id": "allowed", "title": "Allowed", "summary": "Summary."}
        mock_client.return_value.responses.create.return_value = SimpleNamespace(output_text=json.dumps({
            "recommended_record_id": "invented",
            "pitch_script": "Sentence one. Sentence two.",
            "emotional_match_reasons": ["Reason"],
        }))
        with self.assertRaisesRegex(ValueError, "outside the retrieved candidates"):
            generate_final_recommendation("test", [candidate])

    @patch("recommend_and_pitch.get_client")
    def test_rejects_blank_model_reasons(self, mock_client: object) -> None:
        candidate = {"record_id": "allowed", "title": "Allowed", "summary": "Summary."}
        mock_client.return_value.responses.create.return_value = SimpleNamespace(output_text=json.dumps({
            "recommended_record_id": "allowed",
            "pitch_script": "Sentence one. Sentence two.",
            "emotional_match_reasons": ["   "],
        }))
        with self.assertRaisesRegex(ValueError, "no usable emotional"):
            generate_final_recommendation("test", [candidate])

    def test_rejects_more_than_five_rerank_candidates(self) -> None:
        candidates = [{"record_id": str(index), "title": str(index), "summary": "Summary."} for index in range(6)]
        with self.assertRaisesRegex(ValueError, "At most 5"):
            generate_final_recommendation("test", candidates)


class AudioTests(unittest.TestCase):
    def test_rejects_malformed_direct_audio_urls(self) -> None:
        self.assertFalse(is_playable_audio_url("https:///missing-host.mp3"))
        self.assertFalse(is_playable_audio_url("https://example.test/audio.zip"))
        self.assertTrue(is_playable_audio_url("https://example.test/audio.mp3?download=1"))

    def test_validates_summary_speech_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            summary_for_speech({"title": "Test", "summary": "Text"}, max_characters=0)
        with self.assertRaisesRegex(ValueError, "record object"):
            summary_for_speech("not a record")  # type: ignore[arg-type]

    @patch("audio_utils.get_client")
    def test_rejects_invalid_tts_base64(self, mock_client: object) -> None:
        _generate_tts_audio.cache_clear()
        audio_client = mock_client.return_value.with_options.return_value
        audio_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(audio=SimpleNamespace(data="not valid base64!")))]
        )
        with self.assertRaisesRegex(ValueError, "invalid audio data"):
            _generate_tts_audio("A short story summary.")


class PreprocessTests(unittest.TestCase):
    @patch("preprocess_stories.get_client")
    def test_rejects_embedding_batches_with_duplicate_indexes(self, mock_client: object) -> None:
        mock_client.return_value.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[0.0]),
                SimpleNamespace(index=0, embedding=[0.0]),
            ]
        )
        with self.assertRaisesRegex(ValueError, "missing, duplicate, or out-of-order"):
            _embed_payloads(["first", "second"], batch_size=2)

    @patch("preprocess_stories._extract_features")
    def test_retries_a_transient_feature_validation_failure(self, mock_extract: object) -> None:
        expected = {"record_id": "one"}
        mock_extract.side_effect = [ValueError("Feature model returned invalid output."), expected]
        result = _extract_features_with_retry({"title": "One", "summary": "Summary"}, 0)
        self.assertEqual(result, expected)
        self.assertEqual(mock_extract.call_count, 2)


class IntentTests(unittest.TestCase):
    @patch("mood_engine.check_intent.get_client")
    def test_uses_structured_output_and_normalizes_keywords(self, mock_client: object) -> None:
        mock_client.return_value.responses.create.return_value = SimpleNamespace(output_text=json.dumps({
            "intent": "comforting fantasy",
            "keywords": ["Comfort", "MAGIC", "hope"],
            "query": "comforting magical fantasy",
        }))
        result = check_intent("I need a comforting fantasy")

        self.assertEqual(result["keywords"], ["comfort", "magic", "hope"])
        request = mock_client.return_value.responses.create.call_args.kwargs
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertFalse(request["store"])


if __name__ == "__main__":
    unittest.main()
