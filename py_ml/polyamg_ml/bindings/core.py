from __future__ import annotations


class PolyamgCoreUnavailable(RuntimeError):
    pass


def get_core():
    try:
        import _polyamg_core  # type: ignore
    except ImportError as exc:
        raise PolyamgCoreUnavailable(
            "The optional _polyamg_core pybind11 extension is not built. "
            "Use the C++ executables for reproducible runs or build with pybind11 support."
        ) from exc
    return _polyamg_core
