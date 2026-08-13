"""Reproducible price-model report: `python -m travel_intel.ml.report`.

Prints the cross-validated error of the hedonic model against both baselines, plus the
fitted coefficients. Deterministic — same snapshot, same folds, same numbers.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from travel_intel.config import Settings
from travel_intel.domain.enums import Preference
from travel_intel.domain.models import TripRequest
from travel_intel.features.accommodations import build_accommodation_features
from travel_intel.ml.price_model import (
    DistrictMedianBaseline,
    GlobalMedianBaseline,
    HedonicPriceModel,
    cross_validate,
    prepare,
)
from travel_intel.retrieval.snapshot import SnapshotProvider

REFERENCE_REQUEST = TripRequest(
    destination="Tokyo",
    start_date=date(2026, 9, 10),
    end_date=date(2026, 9, 17),
    travelers=2,
    budget_total=2500,
    preferences=(Preference.FOOD, Preference.CULTURE, Preference.NATURE),
)
"""Only the request-independent columns feed the model; this request just builds the table."""


def training_frame(settings: Settings | None = None) -> pd.DataFrame:
    """The full captured market, not the filtered candidate set.

    Candidate generation drops everything above the traveller's budget. Training on what
    survives that filter would truncate the target: the model would never see the expensive
    end of the market and would systematically under-predict what a good hotel costs.
    """
    provider = SnapshotProvider((settings or Settings()).fixtures_dir)
    records = provider.search_accommodations(REFERENCE_REQUEST).records
    return build_accommodation_features(records, REFERENCE_REQUEST)


def main() -> None:
    frame = training_frame()
    prices = frame["price_per_night"]
    print(
        f"Hedonic price model - {len(frame)} properties, "
        f"EUR {prices.min():.0f}-{prices.max():.0f} per night "
        f"(median {prices.median():.0f})\n"
    )

    results = cross_validate(
        frame,
        (GlobalMedianBaseline(), DistrictMedianBaseline(), HedonicPriceModel()),
    )
    fits = next(iter(results.values())).n_fits
    print(f"Repeated 5-fold cross-validation, 5 repeats ({fits} fits per estimator):")
    for name, metrics in results.items():
        print("  " + metrics.as_row(name))

    model = HedonicPriceModel()
    model.fit(prepare(frame))
    print("\nStandardised coefficients (log price, positive = raises the price):")
    for feature, weight in sorted(model.coefficients().items(), key=lambda item: -abs(item[1])):
        print(f"  {feature:<20} {weight:+.4f}")


if __name__ == "__main__":
    main()
