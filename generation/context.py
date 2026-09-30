"""Pack reranked hits into a token-budgeted evidence block with citation ids."""

from dataclasses import dataclass, field

from retrieval.types import Hit, StoredChunk
from generation.prompt import build_evidence_block


@dataclass
class ContextBlock:
    index: int
    hit: Hit
    text: str


@dataclass
class ContextPack:
    blocks: list[ContextBlock] = field(default_factory=list)
    prompt_text: str = ""
    token_count: int = 0


def build_context(
    hits: list[Hit],
    *,
    counter,
    budget: int,
    chunks_by_id: dict[str, StoredChunk] | None = None,
) -> ContextPack:
    """Select hits in order until the token budget is spent.

    Adjacent chunks from the same section are appended when they still fit.
    Duplicate chunk text is skipped. Citation indexes start at 1 and match the
    order of the blocks that were actually packed.
    """
    blocks: list[ContextBlock] = []
    seen: set[str] = set()
    used = 0
    lookup = chunks_by_id or {}
    for hit in hits:
        body = hit.text.strip()
        neighbor_id = (hit.metadata or {}).get("next_chunk_id")
        if not neighbor_id:
            # Stored chunks keep next id only on the chunk, not always on the hit.
            stored = lookup.get(hit.chunk_id)
            if stored is not None:
                neighbor_id = stored.next_chunk_id
        neighbor = lookup.get(neighbor_id) if neighbor_id else None
        if neighbor is not None and neighbor.section and neighbor.section == hit.section:
            body = f"{body}\n{neighbor.text.strip()}"
        key = body.casefold()
        if not body or key in seen:
            continue
        cost = counter.count(body)
        if blocks and used + cost > budget:
            break
        seen.add(key)
        index = len(blocks) + 1
        packed = ContextBlock(index=index, hit=hit, text=body)
        blocks.append(packed)
        used += cost
        if cost > budget:
            break
    prompt = "\n\n".join(build_evidence_block(block.index, _hit_with_text(block)) for block in blocks)
    return ContextPack(blocks=blocks, prompt_text=prompt, token_count=used)


def _hit_with_text(block: ContextBlock) -> Hit:
    hit = block.hit
    if hit.text == block.text:
        return hit
    from retrieval.types import copy_hit

    return copy_hit(hit, text=block.text)
