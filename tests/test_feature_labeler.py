"""Unit tests for feature_labeler module.

Tests the HeuristicFeatureLabeler (deterministic, no deps) and the
OllamaFeatureLabeler with a mocked httpx response.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mech_interp.analysis.feature_labeler import (
    HeuristicFeatureLabeler,
    OllamaFeatureLabeler,
    label_run_features,
)

# ---------------------------------------------------------------------------
# HeuristicFeatureLabeler
# ---------------------------------------------------------------------------


class TestHeuristicFeatureLabeler:
    def test_label_single_clear_prompt(self) -> None:
        labeler = HeuristicFeatureLabeler()
        result = labeler.label(
            feature_index=0,
            top_prompts=["The president signed the treaty"],
            max_activation=3.5,
        )
        # Should extract content words from the prompt
        assert isinstance(result, str)
        assert len(result) > 0
        assert "feature_0" not in result  # not a fallback

    def test_label_multiple_prompts_shared_word(self) -> None:
        labeler = HeuristicFeatureLabeler(top_k=2)
        result = labeler.label(
            feature_index=1,
            top_prompts=[
                "The capital city of France is Paris",
                "The capital city of Germany is Berlin",
                "What is the capital city of Spain",
            ],
            max_activation=4.2,
        )
        assert "capital" in result or "city" in result

    def test_label_empty_prompts_returns_fallback(self) -> None:
        labeler = HeuristicFeatureLabeler()
        result = labeler.label(feature_index=5, top_prompts=[], max_activation=0.0)
        assert "feature_5" in result
        assert "no data" in result

    def test_label_all_stop_words_returns_sparse_fallback(self) -> None:
        labeler = HeuristicFeatureLabeler()
        result = labeler.label(
            feature_index=2,
            top_prompts=["the a an is are was"],
            max_activation=1.0,
        )
        assert "feature_2" in result

    def test_label_deterministic_same_input_same_output(self) -> None:
        labeler = HeuristicFeatureLabeler()
        prompts = ["neural networks learn representations", "deep learning models"]
        r1 = labeler.label(3, prompts, 2.0)
        r2 = labeler.label(3, prompts, 2.0)
        assert r1 == r2

    def test_label_different_features_can_differ(self) -> None:
        labeler = HeuristicFeatureLabeler()
        r1 = labeler.label(0, ["python programming language code"], 1.0)
        r2 = labeler.label(1, ["cooking food recipe kitchen"], 1.0)
        assert r1 != r2

    def test_label_strips_punctuation_from_words(self) -> None:
        labeler = HeuristicFeatureLabeler()
        result = labeler.label(
            feature_index=0,
            top_prompts=["mathematics! algebra, geometry."],
            max_activation=2.0,
        )
        # Should not include punctuation in the label words
        assert "!" not in result
        assert "," not in result


# ---------------------------------------------------------------------------
# OllamaFeatureLabeler (mocked httpx)
# ---------------------------------------------------------------------------


class TestOllamaFeatureLabeler:
    def _mock_response(self, label_text: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": label_text},
            "done": True,
        }
        return mock_resp

    def test_label_returns_model_response(self) -> None:
        labeler = OllamaFeatureLabeler(host="http://localhost:11434", model="llama3.2:3b")
        mock_resp = self._mock_response("mathematical expressions")
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = labeler.label(
                feature_index=0,
                top_prompts=["solve for x in 2x + 3 = 7", "the equation has two solutions"],
                max_activation=5.1,
            )
        assert result == "mathematical expressions"
        mock_post.assert_called_once()

    def test_label_strips_quotes_from_response(self) -> None:
        labeler = OllamaFeatureLabeler()
        with patch("httpx.post", return_value=self._mock_response('"capital cities"')):
            result = labeler.label(0, ["Paris is the capital"], 2.0)
        assert result == "capital cities"

    def test_label_truncates_long_response(self) -> None:
        labeler = OllamaFeatureLabeler()
        long_label = "x" * 200
        with patch("httpx.post", return_value=self._mock_response(long_label)):
            result = labeler.label(0, ["prompt"], 1.0)
        assert len(result) <= 100

    def test_label_returns_fallback_on_http_error(self) -> None:
        labeler = OllamaFeatureLabeler()
        with patch("httpx.post", side_effect=Exception("connection refused")):
            result = labeler.label(feature_index=7, top_prompts=["test"], max_activation=1.0)
        assert "feature_7" in result
        assert "ollama error" in result

    def test_label_sends_correct_model_and_host(self) -> None:
        labeler = OllamaFeatureLabeler(host="http://custom:9999", model="mistral:7b")
        with patch("httpx.post", return_value=self._mock_response("some label")) as mock_post:
            labeler.label(0, ["prompt"], 1.0)
        call_kwargs = mock_post.call_args
        assert "http://custom:9999/api/chat" in call_kwargs[0]
        payload = call_kwargs[1]["json"]
        assert payload["model"] == "mistral:7b"

    def test_label_sends_feature_index_in_user_message(self) -> None:
        labeler = OllamaFeatureLabeler()
        with patch("httpx.post", return_value=self._mock_response("test")) as mock_post:
            labeler.label(feature_index=42, top_prompts=["a prompt"], max_activation=3.0)
        payload = mock_post.call_args[1]["json"]
        user_content = payload["messages"][1]["content"]
        assert "42" in user_content


# ---------------------------------------------------------------------------
# label_run_features integration
# ---------------------------------------------------------------------------


class TestLabelRunFeatures:
    def _write_feature_analysis(self, tmp_path: Path) -> Path:
        data = {
            "n_features": 3,
            "dead_count": 1,
            "live_count": 2,
            "mean_features_per_token": 1.5,
            "features": [
                {
                    "feature_index": 0,
                    "dead": False,
                    "max_activation": 4.2,
                    "mean_activation": 2.1,
                    "top_prompts": [
                        {"rank": 1, "activation": 4.2, "prompt": "The president signed the bill"},
                        {"rank": 2, "activation": 3.1, "prompt": "Congress passed a new law"},
                    ],
                    "coherence_score": 0.45,
                },
                {
                    "feature_index": 1,
                    "dead": False,
                    "max_activation": 2.8,
                    "mean_activation": 1.4,
                    "top_prompts": [
                        {"rank": 1, "activation": 2.8, "prompt": "neural network training"},
                    ],
                    "coherence_score": 0.3,
                },
                {
                    "feature_index": 2,
                    "dead": True,
                    "max_activation": 0.0,
                    "mean_activation": 0.0,
                    "top_prompts": [],
                    "coherence_score": None,
                },
            ],
        }
        path = tmp_path / "feature_analysis.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return tmp_path

    def test_label_run_features_writes_output(self, tmp_path: Path) -> None:
        artifact_dir = self._write_feature_analysis(tmp_path)
        labeler = HeuristicFeatureLabeler()
        labels = label_run_features(artifact_dir, labeler)
        assert isinstance(labels, dict)
        # Dead feature (index 2) should not appear
        assert 2 not in labels
        # Two live features labeled
        assert len(labels) == 2

    def test_label_run_features_writes_json_file(self, tmp_path: Path) -> None:
        artifact_dir = self._write_feature_analysis(tmp_path)
        label_run_features(artifact_dir, HeuristicFeatureLabeler())
        output = tmp_path / "feature_labels.json"
        assert output.is_file()
        data = json.loads(output.read_text())
        assert "feature_labels" in data

    def test_label_run_features_max_features_cap(self, tmp_path: Path) -> None:
        artifact_dir = self._write_feature_analysis(tmp_path)
        labels = label_run_features(artifact_dir, HeuristicFeatureLabeler(), max_features=1)
        # Capped at 1 live feature (highest max_activation)
        assert len(labels) == 1
        # Feature 0 has higher max_activation (4.2 > 2.8)
        assert 0 in labels

    def test_label_run_features_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="feature_analysis.json"):
            label_run_features(tmp_path, HeuristicFeatureLabeler())
