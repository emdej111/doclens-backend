from src.backend.rag import TfidfIndex, chunk_text


def test_chunk_text_respects_size_and_overlap():
    text = "a" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=100)

    assert len(chunks) == 3
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("   ", chunk_size=1000, overlap=100) == []


def test_tfidf_index_ranks_relevant_chunk_first():
    chunks = [
        "The quarterly revenue grew by twelve percent year over year.",
        "The office has a new coffee machine in the kitchen.",
        "Employees can request remote work up to three days per week.",
    ]
    index = TfidfIndex.build(chunks)

    results = index.query("What was the revenue growth?", top_k=1)

    assert len(results) == 1
    chunk_index, score, chunk = results[0]
    assert chunk_index == 0
    assert score > 0


def test_tfidf_index_handles_empty_chunk_list():
    index = TfidfIndex.build([])
    assert index.query("anything", top_k=3) == []
