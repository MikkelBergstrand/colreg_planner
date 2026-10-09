from colregs_planner.nominal.library import LibraryConfig, TrackLibrary
from colregs_planner.nominal.retriever import NominalResult, NominalRetriever

__all__ = [
    "LibraryConfig",
    "TrackLibrary",
    "TrackIndexer",
    "NominalResult",
    "NominalRetriever",
]


def __getattr__(name: str):
    # TrackIndexer builds libraries from raw AIS and needs pandas (the ais
    # extra); import it only when asked for, so using a library does not.
    if name == "TrackIndexer":
        from colregs_planner.nominal.indexer import TrackIndexer
        return TrackIndexer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
