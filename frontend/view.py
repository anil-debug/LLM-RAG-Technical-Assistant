"""Pure helpers for the Streamlit UI.

The page imports Streamlit. These functions do not, so tests can check the
table shape without starting a server.
"""


def hit_rows(hits: list[dict]) -> list[dict]:
    """Columns an operator uses to see why a chunk was kept."""
    rows = []
    for index, hit in enumerate(hits, start=1):
        text = hit.get("text") or ""
        rows.append(
            {
                "rank": index,
                "citation": index,
                "filename": hit.get("filename"),
                "section": hit.get("section"),
                "page": hit.get("page_start"),
                "semantic_score": hit.get("semantic_score"),
                "bm25_score": hit.get("bm25_score"),
                "semantic_rank": hit.get("semantic_rank"),
                "bm25_rank": hit.get("bm25_rank"),
                "fusion_score": hit.get("fusion_score"),
                "rerank_score": hit.get("rerank_score"),
                "preview": text[:240],
            }
        )
    return rows


def citation_rows(citations: list[dict]) -> list[dict]:
    """Map a citation id back to the document, page, section, and chunk."""
    return [
        {
            "id": item.get("index"),
            "filename": item.get("filename"),
            "title": item.get("title"),
            "section": item.get("section"),
            "page": item.get("page_start"),
            "chunk_id": item.get("chunk_id"),
        }
        for item in citations
    ]
