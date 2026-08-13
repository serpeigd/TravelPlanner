"""Feature engineering. One table, two consumers: the ranking and the price model."""

from travel_intel.features.accommodations import (
    MODEL_FEATURES,
    RATING_PRIOR_WEIGHT,
    build_accommodation_features,
)

__all__ = ["MODEL_FEATURES", "RATING_PRIOR_WEIGHT", "build_accommodation_features"]
