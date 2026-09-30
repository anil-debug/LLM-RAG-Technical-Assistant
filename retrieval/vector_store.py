"""In-memory brute-force vectors and a PostgreSQL + pgvector store.

Both stores keep documents, chunks, per-model embeddings, and conversations.
The memory store is the local default and is exact cosine search. PostgreSQL
is the deployment store: embeddings use ``vector(N)`` and an HNSW index per
model name. Two 768-dimensional models share the column. A different
dimension needs a new table, which this build does not create.
"""

import json
import re
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from core.errors import DatabaseUnavailableError, DocumentNotFoundError
from embeddings.embed import l2_normalize
from ingestion.models import Chunk, Document
from retrieval.types import Hit, StoredChunk


class MemoryVectorStore:
    """Process-local store. Safe for the sample corpus and for tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.documents: dict[str, dict] = {}
        self.chunks: dict[str, StoredChunk] = {}
        self.embeddings: dict[str, dict[str, np.ndarray]] = {}
        self.conversations: dict[str, list[dict]] = {}
        self._checksums: dict[str, str] = {}

    def ping(self) -> str:
        return "memory"

    def upsert(
        self,
        document: Document,
        chunks: list[Chunk],
        embeddings: np.ndarray | None,
        model_name: str | None,
    ) -> None:
        with self._lock:
            existing = self._checksums.get(document.checksum)
            if existing and existing != document.id:
                self._delete_unlocked(existing)
            stored = [
                StoredChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    filename=document.filename,
                    title=document.title,
                    text=chunk.text,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    embed_text=chunk.embed_text,
                    token_count=chunk.token_count,
                    prev_chunk_id=chunk.prev_chunk_id,
                    next_chunk_id=chunk.next_chunk_id,
                    metadata={**chunk.metadata, "chunk_index": chunk.chunk_index},
                )
                for chunk in chunks
            ]
            record = _document_record(document, len(chunks))
            record["chunk_ids"] = [chunk.id for chunk in chunks]
            self.documents[document.id] = record
            for chunk in list(self.chunks.values()):
                if chunk.document_id == document.id:
                    self.chunks.pop(chunk.chunk_id, None)
                    for model_vectors in self.embeddings.values():
                        model_vectors.pop(chunk.chunk_id, None)
            for chunk in stored:
                self.chunks[chunk.chunk_id] = chunk
            if embeddings is not None and model_name:
                matrix = l2_normalize(embeddings)
                if len(matrix) != len(stored):
                    raise ValueError("Embedding row count does not match chunk count.")
                bucket = self.embeddings.setdefault(model_name, {})
                for chunk, vector in zip(stored, matrix, strict=True):
                    bucket[chunk.chunk_id] = vector
            self._checksums[document.checksum] = document.id

    def delete(self, document_id: str) -> None:
        with self._lock:
            if document_id not in self.documents:
                raise DocumentNotFoundError(document_id)
            self._delete_unlocked(document_id)

    def _delete_unlocked(self, document_id: str) -> None:
        record = self.documents.pop(document_id, None)
        if record is None:
            return
        self._checksums.pop(record["checksum"], None)
        doomed = [chunk_id for chunk_id, chunk in self.chunks.items() if chunk.document_id == document_id]
        for chunk_id in doomed:
            self.chunks.pop(chunk_id, None)
            for model_vectors in self.embeddings.values():
                model_vectors.pop(chunk_id, None)

    def list_documents(self) -> list[dict]:
        with self._lock:
            return sorted(self.documents.values(), key=lambda item: item["filename"])

    def get_document(self, document_id: str) -> dict:
        with self._lock:
            if document_id not in self.documents:
                raise DocumentNotFoundError(document_id)
            return {"document": self.documents[document_id], "chunks": _ordered_chunks(self, document_id)}

    def all_chunks(self) -> list[StoredChunk]:
        with self._lock:
            return list(self.chunks.values())

    def chunks_by_id(self) -> dict[str, StoredChunk]:
        with self._lock:
            return dict(self.chunks)

    def search_semantic(self, vector: np.ndarray, *, model_name: str, k: int) -> list[Hit]:
        query = l2_normalize(vector)[0]
        with self._lock:
            bucket = self.embeddings.get(model_name, {})
            scored: list[tuple[float, StoredChunk]] = []
            for chunk_id, embedding in bucket.items():
                chunk = self.chunks.get(chunk_id)
                if chunk is None:
                    continue
                scored.append((float(np.dot(query, embedding)), chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            Hit.from_chunk(chunk, semantic_score=score, semantic_rank=rank)
            for rank, (score, chunk) in enumerate(scored[:k], start=1)
        ]

    def ensure_conversation(self, conversation_id: str | None) -> str:
        with self._lock:
            if conversation_id and conversation_id in self.conversations:
                return conversation_id
            new_id = conversation_id or str(uuid4())
            self.conversations.setdefault(new_id, [])
            return new_id

    def add_message(self, conversation_id: str, role: str, content: str, citations: list[dict]) -> None:
        with self._lock:
            self.conversations.setdefault(conversation_id, []).append(
                {
                    "role": role,
                    "content": content,
                    "citations": citations,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )

    def list_messages(self, conversation_id: str) -> list[dict]:
        with self._lock:
            return list(self.conversations.get(conversation_id, []))


def _ordered_chunks(store: MemoryVectorStore, document_id: str) -> list[dict]:
    chunks = [chunk for chunk in store.chunks.values() if chunk.document_id == document_id]
    chunks.sort(key=lambda chunk: chunk.chunk_id)
    record = store.documents[document_id]
    order = record.get("chunk_ids") or []
    if order:
        position = {chunk_id: index for index, chunk_id in enumerate(order)}
        chunks.sort(key=lambda chunk: position.get(chunk.chunk_id, 10**9))
    return [_chunk_dict(chunk) for chunk in chunks]


def _document_record(document: Document, chunk_count: int) -> dict:
    return {
        "id": document.id,
        "filename": document.filename,
        "source": document.source,
        "media_type": document.media_type,
        "checksum": document.checksum,
        "title": document.title,
        "created_at": document.created_at.isoformat(),
        "metadata": document.metadata,
        "chunk_count": chunk_count,
        "chunk_ids": [],
    }


def _chunk_dict(chunk: StoredChunk) -> dict:
    return {
        "id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "section": chunk.section,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata,
    }


def _vector_literal(vector: np.ndarray) -> str:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


class PostgresVectorStore:
    """PostgreSQL store using pgvector cosine distance and per-model HNSW indexes."""

    def __init__(self, database_url: str, dimension: int = 768) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        self.database_url = database_url
        self.dimension = int(dimension)
        self._schema_ready = False
        self._indexes: set[str] = set()
        self._lock = threading.Lock()

    def ping(self) -> str:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return "postgres"

    def ensure_schema(self) -> None:
        with self._lock:
            if self._schema_ready:
                return
            dim = str(self.dimension)
            statements = [
                "CREATE EXTENSION IF NOT EXISTS vector",
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    checksum TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embed_text TEXT NOT NULL,
                    section TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    token_count INTEGER NOT NULL,
                    prev_chunk_id TEXT,
                    next_chunk_id TEXT,
                    metadata JSONB NOT NULL
                )
                """,
                f"""
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model_name TEXT NOT NULL,
                    embedding vector({dim}) NOT NULL,
                    PRIMARY KEY (chunk_id, model_name)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """,
            ]
            with self._connect() as conn:
                with conn.cursor() as cur:
                    for statement in statements:
                        cur.execute(statement)
            self._schema_ready = True

    def ensure_model_index(self, model_name: str) -> None:
        self.ensure_schema()
        if model_name in self._indexes:
            return
        if not re.fullmatch(r"[A-Za-z0-9_./:+-]+", model_name):
            raise ValueError(f"Refusing to interpolate model name {model_name!r} into SQL.")
        slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")[:40]
        literal = model_name.replace("'", "''")
        statement = (
            f"CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_{slug} "
            f"ON chunk_embeddings USING hnsw (embedding vector_cosine_ops) "
            f"WHERE model_name = '{literal}'"
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(statement)
        self._indexes.add(model_name)

    def upsert(
        self,
        document: Document,
        chunks: list[Chunk],
        embeddings: np.ndarray | None,
        model_name: str | None,
    ) -> None:
        self.ensure_schema()
        if embeddings is not None and model_name:
            self.ensure_model_index(model_name)
            matrix = l2_normalize(embeddings)
            if len(matrix) != len(chunks):
                raise ValueError("Embedding row count does not match chunk count.")
            if matrix.shape[1] != self.dimension:
                raise ValueError(
                    f"Embedding dimension {matrix.shape[1]} does not match vector({self.dimension})."
                )
        else:
            matrix = None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM documents WHERE checksum = %s", (document.checksum,))
                row = cur.fetchone()
                if row and row[0] != document.id:
                    cur.execute("DELETE FROM documents WHERE id = %s", (row[0],))
                cur.execute("DELETE FROM documents WHERE id = %s", (document.id,))
                cur.execute(
                    """
                    INSERT INTO documents
                        (id, filename, source, media_type, checksum, title, body, created_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        document.id,
                        document.filename,
                        document.source,
                        document.media_type,
                        document.checksum,
                        document.title,
                        document.text,
                        document.created_at,
                        json.dumps(document.metadata),
                    ),
                )
                for chunk in chunks:
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            id, document_id, chunk_index, content, embed_text, section,
                            page_start, page_end, token_count, prev_chunk_id, next_chunk_id, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            chunk.id,
                            document.id,
                            chunk.chunk_index,
                            chunk.text,
                            chunk.embed_text,
                            chunk.section,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.token_count,
                            chunk.prev_chunk_id,
                            chunk.next_chunk_id,
                            json.dumps({**chunk.metadata, "chunk_index": chunk.chunk_index}),
                        ),
                    )
                if matrix is not None and model_name:
                    for chunk, vector in zip(chunks, matrix, strict=True):
                        cur.execute(
                            """
                            INSERT INTO chunk_embeddings (chunk_id, model_name, embedding)
                            VALUES (%s, %s, %s::vector)
                            """,
                            (chunk.id, model_name, _vector_literal(vector)),
                        )

    def delete(self, document_id: str) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
                if cur.rowcount == 0:
                    raise DocumentNotFoundError(document_id)

    def list_documents(self) -> list[dict]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT d.id, d.filename, d.source, d.media_type, d.checksum, d.title,
                           d.created_at, d.metadata, COUNT(c.id)
                    FROM documents d
                    LEFT JOIN chunks c ON c.document_id = d.id
                    GROUP BY d.id
                    ORDER BY d.filename
                    """
                )
                rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "filename": row[1],
                "source": row[2],
                "media_type": row[3],
                "checksum": row[4],
                "title": row[5],
                "created_at": row[6].isoformat(),
                "metadata": row[7],
                "chunk_count": row[8],
            }
            for row in rows
        ]

    def get_document(self, document_id: str) -> dict:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, source, media_type, checksum, title, created_at, metadata
                    FROM documents WHERE id = %s
                    """,
                    (document_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise DocumentNotFoundError(document_id)
                cur.execute(
                    """
                    SELECT id, document_id, content, section, page_start, page_end, token_count, metadata
                    FROM chunks WHERE document_id = %s ORDER BY chunk_index
                    """,
                    (document_id,),
                )
                chunks = cur.fetchall()
        return {
            "document": {
                "id": row[0],
                "filename": row[1],
                "source": row[2],
                "media_type": row[3],
                "checksum": row[4],
                "title": row[5],
                "created_at": row[6].isoformat(),
                "metadata": row[7],
            },
            "chunks": [
                {
                    "id": chunk[0],
                    "document_id": chunk[1],
                    "text": chunk[2],
                    "section": chunk[3],
                    "page_start": chunk[4],
                    "page_end": chunk[5],
                    "token_count": chunk[6],
                    "metadata": chunk[7],
                }
                for chunk in chunks
            ],
        }

    def all_chunks(self) -> list[StoredChunk]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.document_id, d.filename, d.title, c.content, c.section,
                           c.page_start, c.page_end, c.embed_text, c.token_count,
                           c.prev_chunk_id, c.next_chunk_id, c.metadata
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    ORDER BY d.filename, c.chunk_index
                    """
                )
                rows = cur.fetchall()
        return [
            StoredChunk(
                chunk_id=row[0],
                document_id=row[1],
                filename=row[2],
                title=row[3],
                text=row[4],
                section=row[5],
                page_start=row[6],
                page_end=row[7],
                embed_text=row[8],
                token_count=row[9],
                prev_chunk_id=row[10],
                next_chunk_id=row[11],
                metadata=row[12] or {},
            )
            for row in rows
        ]

    def chunks_by_id(self) -> dict[str, StoredChunk]:
        return {chunk.chunk_id: chunk for chunk in self.all_chunks()}

    def search_semantic(self, vector: np.ndarray, *, model_name: str, k: int) -> list[Hit]:
        self.ensure_schema()
        literal = _vector_literal(l2_normalize(vector)[0])
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.document_id, d.filename, d.title, c.content, c.section,
                           c.page_start, c.page_end, c.metadata,
                           1 - (e.embedding <=> %s::vector) AS semantic_score
                    FROM chunk_embeddings e
                    JOIN chunks c ON c.id = e.chunk_id
                    JOIN documents d ON d.id = c.document_id
                    WHERE e.model_name = %s
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (literal, model_name, literal, k),
                )
                rows = cur.fetchall()
        return [
            Hit(
                chunk_id=row[0],
                document_id=row[1],
                filename=row[2],
                title=row[3],
                text=row[4],
                section=row[5],
                page_start=row[6],
                page_end=row[7],
                metadata=row[8] or {},
                semantic_score=float(row[9]),
                semantic_rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def ensure_conversation(self, conversation_id: str | None) -> str:
        self.ensure_schema()
        new_id = conversation_id or str(uuid4())
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (id, created_at)
                    VALUES (%s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (new_id, datetime.now(UTC)),
                )
        return new_id

    def add_message(self, conversation_id: str, role: str, content: str, citations: list[dict]) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (id, conversation_id, role, content, citations, created_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (str(uuid4()), conversation_id, role, content, json.dumps(citations), datetime.now(UTC)),
                )

    def list_messages(self, conversation_id: str) -> list[dict]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content, citations, created_at
                    FROM messages WHERE conversation_id = %s ORDER BY created_at, id
                    """,
                    (conversation_id,),
                )
                rows = cur.fetchall()
        return [
            {
                "role": row[0],
                "content": row[1],
                "citations": row[2] or [],
                "created_at": row[3].isoformat(),
            }
            for row in rows
        ]

    @contextmanager
    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise DatabaseUnavailableError("psycopg is not installed.") from exc
        try:
            connection = psycopg.connect(self.database_url)
        except Exception as exc:
            raise DatabaseUnavailableError(f"Could not connect to PostgreSQL: {exc}") from exc
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

