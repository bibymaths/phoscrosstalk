from types import SimpleNamespace

def namespace_to_dict(x):
    """Convert config namespace/table objects to a plain dict."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, SimpleNamespace):
        return vars(x)
    if hasattr(x, "__dict__"):
        return vars(x)
    raise TypeError(
        f"optimizer_backend_kwargs must be a table/dict-like object, got {type(x).__name__}"
    )