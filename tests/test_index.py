import numpy as np
import pytest

from rag.chunking import Chunk
from rag.index import VectorIndex, reciprocal_rank_fusion


def _chunk(chunk_id: str, doc_id: str, lang: str, text: str = "x") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=f"{doc_id}.txt",
        lang=lang,
        index=0,
        text=text,
    )


# ---------- RRF ----------


def test_rrf_prefers_items_ranked_high_in_multiple_lists():
    rankings = {"zh": ["a", "b", "c"], "en": ["b", "d"]}
    ordered = [chunk_id for chunk_id, _ in reciprocal_rank_fusion(rankings)]
    assert ordered[0] == "b"  # 两路都靠前
    assert ordered.index("c") > ordered.index("d")  # c 只在 zh 的末位,d 在 en 的次位


def test_rrf_scores_are_symmetric_between_equal_rankings():
    rankings = {"zh": ["a"], "en": ["b"]}
    scores = dict(reciprocal_rank_fusion(rankings))
    assert scores["a"] == pytest.approx(scores["b"])


def test_rrf_weights_favour_the_weighted_space():
    rankings = {"zh": ["a"], "en": ["b"]}
    scores = dict(reciprocal_rank_fusion(rankings, weights={"zh": 2.0}))
    assert scores["a"] > scores["b"]


def test_rrf_tie_break_is_deterministic():
    rankings = {"zh": ["b", "a"], "en": ["a", "b"]}
    assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(rankings)


# ---------- VectorIndex ----------


def test_search_returns_nearest_neighbour():
    index = VectorIndex(("zh",))
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    index.add("zh", vectors, [_chunk(f"d{i}#000", f"d{i}", "zh") for i in range(3)])
    index.build()
    hits = index.search("zh", np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert hits[0].chunk_id == "d0#000"
    assert len(hits) == 2


def test_spaces_are_isolated():
    index = VectorIndex(("zh", "en"))
    index.add(
        "zh",
        np.eye(2, dtype=np.float32),
        [_chunk("zh#000", "zhd", "zh"), _chunk("zh#001", "zhd", "zh")],
    )
    index.add(
        "en",
        np.eye(2, dtype=np.float32),
        [_chunk("en#000", "end", "en"), _chunk("en#001", "end", "en")],
    )
    index.build()
    hits = index.search("zh", np.array([1.0, 0.0], dtype=np.float32), k=5)
    assert {hit.lang for hit in hits} == {"zh"}


def test_search_on_unbuilt_space_returns_empty():
    index = VectorIndex(("zh", "en"))
    index.add("zh", np.ones((1, 2), dtype=np.float32), [_chunk("zh#000", "zhd", "zh")])
    index.build()
    assert index.search("en", np.array([1.0, 0.0], dtype=np.float32), k=3) == []


def test_vector_count_must_match_chunk_count():
    index = VectorIndex(("zh",))
    with pytest.raises(ValueError):
        index.add("zh", np.zeros((2, 2), dtype=np.float32), [_chunk("a#000", "a", "zh")])


def test_unknown_space_is_rejected():
    index = VectorIndex(("zh",))
    with pytest.raises(ValueError):
        index.add("en", np.zeros((1, 2), dtype=np.float32), [_chunk("a#000", "a", "en")])


def test_save_and_load_roundtrip(tmp_path):
    index = VectorIndex(("zh",))
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index.add(
        "zh",
        vectors,
        [_chunk("a#000", "a", "zh", "alpha"), _chunk("a#001", "a", "zh", "beta")],
    )
    index.build()
    query = np.array([1.0, 0.0], dtype=np.float32)
    before = index.search("zh", query, k=2)

    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    after = loaded.search("zh", query, k=2)

    assert [hit.chunk_id for hit in before] == [hit.chunk_id for hit in after]
    assert [hit.text for hit in after] == ["alpha", "beta"]
    assert loaded.stats() == {"zh": 2}
