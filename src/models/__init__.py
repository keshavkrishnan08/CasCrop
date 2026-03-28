"""CasCrop model architectures.

Full model:
    CasCrop — cascading crop-waste prediction with ECMP graph layers.

Baselines (ablation rows):
    Row 1: LocalOnlyModel      — MLP on biophysical features only
    Row 2: LocalEconModel      — MLP on bio + econ features, no graph
    Row 3: GeoGATModel         — standard GAT on geographic adjacency
    Row 4: SymmetricECMPModel  — ECMP with symmetric shock embedding
"""

from .cascrop import CasCrop

from .encoders import (
    BiophysicalEncoder,
    EconomicEncoder,
    DisentanglementModule,
    GradientReversalLayer,
)
from .graph import (
    ECMPLayer,
    ECMPStack,
    StandardGATLayer,
    GATStack,
    DynamicGraphConstructor,
)
from .heads import (
    WasteClassifier,
    CauseClassifier,
    CauseOfLoss,
)
from .baselines import (
    LocalOnlyModel,
    LocalEconModel,
    GeoGATModel,
    SymmetricECMPModel,
)

__all__ = [
    # Full model
    "CasCrop",
    # Encoders
    "BiophysicalEncoder",
    "EconomicEncoder",
    "DisentanglementModule",
    "GradientReversalLayer",
    # Graph
    "ECMPLayer",
    "ECMPStack",
    "StandardGATLayer",
    "GATStack",
    "DynamicGraphConstructor",
    # Heads
    "WasteClassifier",
    "CauseClassifier",
    "CauseOfLoss",
    # Baselines
    "LocalOnlyModel",
    "LocalEconModel",
    "GeoGATModel",
    "SymmetricECMPModel",
]
