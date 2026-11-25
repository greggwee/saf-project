"""Canonical feature names and feature sets (BLT Table 2).

Suffix ``_neg`` marks negative realized semicorrelation features (RC^-),
horizons are ``d`` (day), ``w`` (week), ``m`` (month), ``q`` (quarter, CoM = 63).
"""

from __future__ import annotations

HAR_HORIZONS = ("d", "w", "m")
EXP_HORIZONS = ("d", "w", "m", "q")

RC = tuple(f"RC_{h}" for h in HAR_HORIZONS)
RC_NEG = tuple(f"RC_{h}_neg" for h in HAR_HORIZONS)
FRC = tuple(f"FRC_{h}" for h in HAR_HORIZONS)
EXP_RC = tuple(f"ExpRC_{h}" for h in EXP_HORIZONS)
EXP_RC_NEG = tuple(f"ExpRC_{h}_neg" for h in EXP_HORIZONS)
EXP_SC_RC = tuple(f"ExpScRC_{h}" for h in EXP_HORIZONS)
EXP_SC_RC_NEG = tuple(f"ExpScRC_{h}_neg" for h in EXP_HORIZONS)

ALL_FEATURES: tuple[str, ...] = RC + RC_NEG + FRC + EXP_RC + EXP_RC_NEG + EXP_SC_RC + EXP_SC_RC_NEG

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "har": RC,
    "shar": RC + RC_NEG,
    "shar_f": RC + RC_NEG + FRC,
    "all": ALL_FEATURES,
}

TARGET = "target"


def resolve_features(spec: str | list[str]) -> tuple[str, ...]:
    if isinstance(spec, str):
        if spec not in FEATURE_SETS:
            raise KeyError(f"unknown feature set {spec!r}; known: {sorted(FEATURE_SETS)}")
        return FEATURE_SETS[spec]
    unknown = set(spec) - set(ALL_FEATURES)
    if unknown:
        raise KeyError(f"unknown features {sorted(unknown)}")
    return tuple(spec)
