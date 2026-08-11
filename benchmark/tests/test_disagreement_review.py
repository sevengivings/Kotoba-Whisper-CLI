from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "disagreement_review.py"
SPEC = importlib.util.spec_from_file_location("benchmark_disagreement_review", MODULE_PATH)
assert SPEC and SPEC.loader
review = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review
SPEC.loader.exec_module(review)


def test_disagreement_candidates_include_low_similarity_overlap() -> None:
    clip = review.Clip("R03", Path("R03.wav"), 30.0, True, [])
    candidates = review.disagreement_candidates(
        clip,
        [review.Chunk(0.0, 4.0, "今の気持ちを率直に送らせてください")],
        [review.Chunk(0.0, 4.5, "今の気持ちを素直にお聞かせください")],
        similarity_threshold=0.9,
        min_overlap_s=0.2,
        padding_s=0.5,
        min_window_s=5.0,
        max_window_s=15.0,
    )

    assert len(candidates) == 1
    assert candidates[0].reason == "low_similarity"
    assert 5.0 <= candidates[0].end - candidates[0].start <= 15.0


def test_disagreement_candidates_skip_high_similarity_overlap() -> None:
    clip = review.Clip("R03", Path("R03.wav"), 30.0, True, [])
    candidates = review.disagreement_candidates(
        clip,
        [review.Chunk(0.0, 4.0, "さあそして")],
        [review.Chunk(0.0, 4.0, "さあ、そして。")],
        similarity_threshold=0.9,
        min_overlap_s=0.2,
        padding_s=0.5,
        min_window_s=5.0,
        max_window_s=15.0,
    )

    assert candidates == []


def test_chunks_from_srt_parses_basic_srt(tmp_path: Path) -> None:
    path = tmp_path / "R01.ja.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,500\nこんにちは\n", encoding="utf-8")

    chunks = review.chunks_from_srt(path)

    assert chunks == [review.Chunk(1.0, 2.5, "こんにちは")]
