"""Accommodation feature table.

One builder feeds both consumers: the ranking (which turns these into [0, 1] factors) and
the price model in `ml/` (which trains on the request-independent subset, `MODEL_FEATURES`).
Two builders would drift apart, and the day they did, the model would be scoring options on
features the ranking no longer computes the same way.

Missing stays missing here. Nothing is imputed at this layer — `NaN` reaches the scorer,
which drops the factor and redistributes its weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from travel_intel.budget import DEFAULT_BUDGET_POLICY, BudgetPolicy
from travel_intel.domain.enums import BudgetCategory, Preference
from travel_intel.domain.models import Accommodation, TripRequest

RATING_PRIOR_WEIGHT = 200.0
"""Strength of the prior in the Bayesian shrinkage of guest ratings, in reviews.

A 9.2 from 136 reviews and an 8.6 from 3,213 are not comparable: the first is mostly noise.
Each rating is pulled toward the market mean with weight `m`, so a property needs roughly
`m` reviews before its own score dominates. 200 sits at the knee of this dataset's review
distribution — most serious hotels clear it, small and brand-new listings do not, which is
exactly the distinction we want the ranking to make.
"""

AMENITY_FLAGS: dict[str, str] = {
    "has_restaurant": "Restaurant",
    "has_bar": "Bar",
    "has_room_service": "Room service",
    "has_spa": "Spa and wellness center",
    "has_pool": "Swimming pool",
    "has_fitness": "Fitness center",
    "has_public_bath": "Public Bath",
    "has_garden": "Garden",
    "has_parking": "Parking",
    "has_front_desk_24h": "24-hour front desk",
    "has_free_wifi": "Free Wifi",
}
"""Amenity indicators kept as columns: present in enough properties to carry signal, absent
in enough to discriminate. Universal ones (smoke alarms) and near-unique ones are useless."""

PREFERENCE_AMENITIES: dict[Preference, frozenset[str]] = {
    Preference.FOOD: frozenset({"Restaurant", "Bar", "Room service"}),
    Preference.NATURE: frozenset({"Garden", "Terrace"}),
    Preference.RELAXATION: frozenset(
        {"Spa and wellness center", "Sauna", "Hot tub/Jacuzzi", "Public Bath", "Swimming pool"}
    ),
    Preference.NIGHTLIFE: frozenset({"Bar"}),
}
"""Preferences for which a *property* carries evidence.

CULTURE, ART, HISTORY and SHOPPING are deliberately absent. A hotel is not more or less
cultural, and pretending otherwise would manufacture a signal out of nothing. Those
preferences are served by the itinerary, which is where they are actually matched — see
`preference_match` below for how their absence is handled.
"""

MODEL_FEATURES: tuple[str, ...] = (
    "stars",
    "rating",
    "review_count",
    "distance_km",
    "n_amenities",
    *AMENITY_FLAGS,
)
"""Request-independent columns: the only ones the price model may train on.

Anything derived from the request (budget ratio, preference match) would leak the query into
a model that is supposed to describe the *property*.
"""


def _preference_match(amenities: frozenset[str], preferences: tuple[Preference, ...]) -> float:
    """Share of the user's evidence-bearing preferences this property supports.

    Preferences with no property-level evidence are excluded from the denominator rather
    than counted as unmatched: a hotel should not lose points for failing to be a museum.
    When none of the user's preferences carry property-level evidence, the result is `NaN`
    and the scorer drops the factor entirely.
    """
    with_evidence = [p for p in preferences if PREFERENCE_AMENITIES.get(p)]
    if not with_evidence:
        return float("nan")
    matched = sum(bool(amenities & PREFERENCE_AMENITIES[p]) for p in with_evidence)
    return matched / len(with_evidence)


def build_accommodation_features(
    records: tuple[Accommodation, ...],
    request: TripRequest,
    policy: BudgetPolicy = DEFAULT_BUDGET_POLICY,
) -> pd.DataFrame:
    """Build the feature table for a candidate set, indexed by accommodation id."""
    if not records:
        raise ValueError("cannot build features for an empty candidate set")

    allowance = policy.allowance(BudgetCategory.ACCOMMODATION, request.budget_total)
    rows: list[dict[str, object]] = []
    for record in records:
        amenities = frozenset(record.amenities)
        rows.append(
            {
                "id": record.id,
                "name": record.name,
                "district": record.neighborhood,
                "price_per_night": record.price_per_night,
                "total_cost": record.total_cost(request.nights),
                "stars": record.stars,
                "rating": record.rating,
                "review_count": record.review_count,
                "distance_km": record.distance_to_center_km,
                "completeness": record.data_completeness,
                "n_amenities": len(amenities),
                "preference_match": _preference_match(amenities, request.preferences),
                **{column: amenities.issuperset({name}) for column, name in AMENITY_FLAGS.items()},
            }
        )

    frame = pd.DataFrame(rows).set_index("id")
    frame["budget_ratio"] = frame["total_cost"] / allowance
    frame["rating_shrunk"] = _shrink_ratings(frame["rating"], frame["review_count"])
    return frame


def _shrink_ratings(ratings: pd.Series, review_counts: pd.Series) -> pd.Series:
    """Pull each rating toward the market mean in proportion to how thin its evidence is.

    The market mean is computed over this candidate set, so the comparison is always against
    the alternatives actually on the table. Properties with no rating stay `NaN`: shrinkage
    sharpens weak evidence, it does not invent it.
    """
    market_mean = ratings.mean(skipna=True)
    if pd.isna(market_mean):
        return pd.Series(np.nan, index=ratings.index, dtype="float64")
    weights = review_counts.fillna(0.0).astype("float64")
    shrunk = (ratings * weights + market_mean * RATING_PRIOR_WEIGHT) / (
        weights + RATING_PRIOR_WEIGHT
    )
    return shrunk.where(ratings.notna())
