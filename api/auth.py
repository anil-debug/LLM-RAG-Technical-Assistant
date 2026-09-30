"""Authentication extension point.

The default principal is anonymous. Replace ``get_principal`` with a
dependency that validates a credential when you expose this service beyond
a trusted network. Routes already depend on it, so the check has one place
to land.
"""

from pydantic import BaseModel


class Principal(BaseModel):
    subject: str = "anonymous"
    scopes: list[str] = []


def get_principal() -> Principal:
    """Return the caller. The default implementation does not enforce auth."""
    return Principal()
