"""colregs_planner: COLREGs-aware multi-vessel trajectory prediction and
own-ship manoeuvre planning.

    from colregs_planner import ColregsDPPlanner, Route      # DP reference for an own ship
    from colregs_planner import PredictionService            # predictions for a planner
"""

from importlib.metadata import PackageNotFoundError, version

from colregs_planner.planner_api import (
    ColregsDPPlanner,
    DPReference,
    PredictionService,
    Route,
)

try:
    __version__ = version("colregs-planner")
except PackageNotFoundError:          # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ["ColregsDPPlanner", "DPReference", "PredictionService", "Route", "__version__"]
