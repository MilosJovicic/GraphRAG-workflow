from pathlib import Path


SRC_DIR = Path(__file__).parent.parent / "src"


def test_per_type_validates_embedding_count_before_assignment():
    source = (SRC_DIR / "workflows" / "per_type.py").read_text(encoding="utf-8")

    assert "len(embeddings) != len(to_embed_indices)" in source
    assert "Embedding count mismatch" in source


def test_per_type_writes_all_successful_rows_without_embedding_filter():
    source = (SRC_DIR / "workflows" / "per_type.py").read_text(encoding="utf-8")

    assert 'if r["embedding"] is not None' not in source
    assert "args=[rows_to_write]" in source
