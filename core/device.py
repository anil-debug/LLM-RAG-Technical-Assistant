"""Select a torch device without silently ignoring a CUDA request."""

from core.errors import ModelUnavailableError


def resolve_device(requested: str) -> str:
    """Return ``cpu`` or ``cuda``.

    A request for CUDA fails when this process cannot see a GPU. Falling
    through to CPU would hide a configuration mistake and change latency.
    """
    if requested == "cpu":
        return "cpu"
    if requested != "cuda":
        raise ModelUnavailableError(f"Unsupported device '{requested}'. Use cpu or cuda.")
    import torch

    if not torch.cuda.is_available():
        raise ModelUnavailableError("CUDA was requested but no GPU is available to PyTorch.")
    return "cuda"
