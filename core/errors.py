"""Application errors mapped to HTTP responses at the API boundary."""


class AppError(Exception):
    """Base class for errors the application knows how to explain."""


class DocumentNotFoundError(AppError):
    """No document exists for the requested id."""


class UnsupportedMediaTypeError(AppError):
    """The upload is not a PDF, Markdown, text, HTML, or log file."""


class EmptyDocumentError(AppError):
    """The upload contained no bytes."""


class UploadTooLargeError(AppError):
    """The upload exceeds the configured byte limit."""


class ModelUnavailableError(AppError):
    """A configured model or LLM server cannot be used right now."""


class TokenizerUnavailableError(AppError):
    """The embedding tokenizer could not be loaded."""


class DatabaseUnavailableError(AppError):
    """PostgreSQL could not be reached or initialized."""


class ClassifierNotReadyError(AppError):
    """A document-intelligence classifier was enabled without trained weights."""
