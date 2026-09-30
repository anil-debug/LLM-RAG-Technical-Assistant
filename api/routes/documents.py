"""Document upload, list, and delete."""

from fastapi import APIRouter, Depends, File, Request, UploadFile

from api.auth import Principal, get_principal
from api.schemas.models import DocumentCreated, DocumentSummary
from api.service import save_upload
from core.errors import DocumentNotFoundError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentCreated)
async def create_document(
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(get_principal),
) -> DocumentCreated:
    del principal
    data = await file.read()
    created = save_upload(request.app.state.container, file.filename or "upload", data)
    return DocumentCreated(**created)


@router.get("", response_model=list[DocumentSummary])
def list_documents(
    request: Request,
    principal: Principal = Depends(get_principal),
) -> list[DocumentSummary]:
    del principal
    rows = request.app.state.container.store.list_documents()
    return [
        DocumentSummary(
            id=row["id"],
            filename=row["filename"],
            media_type=row["media_type"],
            title=row["title"],
            checksum=row["checksum"],
            chunk_count=row["chunk_count"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


@router.get("/{document_id}")
def get_document(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> dict:
    del principal
    return request.app.state.container.store.get_document(document_id)


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> dict:
    del principal
    try:
        request.app.state.container.store.delete(document_id)
    except DocumentNotFoundError:
        raise
    return {"deleted": document_id}
