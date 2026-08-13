"""Hedonic price model.

**What it predicts and why that is not the product.** The model regresses a property's
nightly price on the property's own attributes — stars, guest rating, review volume,
distance to the centre, how many amenities it offers. The prediction itself is worthless to
a traveller: nobody needs an estimate of a price that is printed on the page. The
**residual** is the product. A room priced well below what its own features predict is good
value *for what it is*; one priced well above is not. That residual becomes the
`value_for_money` ranking factor.

This is a hedonic regression in the economics sense — the same construction used for
house-price and CPI quality adjustment — and it is the reason the model is fitted on the
candidate set itself. "Cheap for what it is" is a statement about a market, so the market it
is measured against is the set of options actually on the table. Residuals are therefore
**in-sample by design**; the model's ability to generalise to unseen properties is a
different question, and it is measured separately by `cross_validate`.

**Why price and not relevance.** Price is the one target that genuinely exists in this data.
Relevance labels — clicks, bookings — do not, and fabricating them would train a model to
recover the heuristic used to fabricate them. See ADR-003.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "price_per_night"

PRICE_FEATURES: tuple[str, ...] = (
    "stars",
    "rating",
    "log_review_count",
    "distance_km",
    "n_amenities",
)
"""Deliberately five features for a few dozen rows.

Every column in `MODEL_FEATURES` was available, including eleven amenity indicators. They
are collapsed into `n_amenities` instead: with this sample size, one parameter per amenity
would fit noise and produce coefficients that look meaningful and are not. Knowing when the
data cannot support the model you would like is the point.
"""

RANDOM_STATE = 42
CV_SPLITS = 5
CV_REPEATS = 5


@dataclass(frozen=True)
class PriceMetrics:
    """Out-of-sample error, in euros per night, on the original (not log) scale."""

    mae: float
    rmse: float
    mape: float
    n_observations: int
    n_fits: int

    def as_row(self, name: str) -> str:
        return f"{name:<24} MAE {self.mae:7.2f}  RMSE {self.rmse:7.2f}  MAPE {self.mape:6.1%}"


class PriceEstimator(Protocol):
    """Anything that can learn a price from a feature frame and predict one back."""

    name: str

    def fit(self, frame: pd.DataFrame) -> None: ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the model's input columns from the shared feature table.

    `review_count` is log-transformed because its distribution spans four orders of
    magnitude (1 to 23,178); untransformed, a single mega-chain hotel would dominate every
    coefficient.
    """
    prepared = frame.copy()
    prepared["log_review_count"] = np.log1p(prepared["review_count"].astype("float64"))
    return prepared


class GlobalMedianBaseline:
    """Predict the same price for everything. The floor any model must clear."""

    name = "baseline: global median"

    def __init__(self) -> None:
        self._median: float = 0.0

    def fit(self, frame: pd.DataFrame) -> None:
        self._median = float(frame[TARGET].median())

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self._median, dtype="float64")


class DistrictMedianBaseline:
    """Predict the median price of the property's district.

    The honest competitor. Location explains a great deal of price in any city, and a
    grouped median needs no model at all — so a regression that cannot beat this has not
    earned its place.
    """

    name = "baseline: district median"

    def __init__(self) -> None:
        self._by_district: dict[str, float] = {}
        self._fallback: float = 0.0

    def fit(self, frame: pd.DataFrame) -> None:
        self._fallback = float(frame[TARGET].median())
        self._by_district = {
            str(district): float(group[TARGET].median())
            for district, group in frame.groupby("district", dropna=True)
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self._by_district.get(str(d), self._fallback) for d in frame["district"]],
            dtype="float64",
        )


class HedonicPriceModel:
    """Ridge regression on log price over a handful of property attributes.

    Three modelling choices, each for a stated reason:

    - **Log target.** Nightly prices here span €70-€646 and are right-skewed. On the raw
      scale, errors on expensive hotels would dominate the fit and the model would predict
      the luxury tail well and everything else badly. In log space the model learns
      *proportional* differences, which is also how the value residual is interpreted.
    - **Ridge, not OLS.** Stars, rating and amenity count are correlated; regularisation
      keeps coefficients stable rather than letting collinearity throw them around.
    - **Median imputation for missing features.** One property has no stars and no rating.
      Imputing a *feature* inside a fitted model, with a documented strategy learned on the
      training fold only, is standard practice — and quite different from imputing a
      *score*, which the ranking refuses to do because it would fabricate a judgement.
    """

    name = "hedonic ridge (log price)"

    def __init__(self, alpha: float = 1.0) -> None:
        self._pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=None)),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> None:
        self._pipeline.fit(frame[list(PRICE_FEATURES)], np.log(frame[TARGET].to_numpy()))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predicted_log = self._pipeline.predict(frame[list(PRICE_FEATURES)])
        return np.exp(np.asarray(predicted_log, dtype="float64"))

    def fit_predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Fit on the candidate set and predict it back — the hedonic use, in-sample by design.

        Generalisation is a separate question, answered by `cross_validate`, not by this.
        """
        prepared = prepare(frame)
        self.fit(prepared)
        return self.predict(prepared)

    def coefficients(self) -> dict[str, float]:
        """Standardised coefficients, in log-price units. Positive means 'raises the price'."""
        ridge: Ridge = self._pipeline.named_steps["ridge"]
        return {
            feature: round(float(weight), 4)
            for feature, weight in zip(PRICE_FEATURES, ridge.coef_, strict=True)
        }


def _metrics(actual: np.ndarray, predicted: np.ndarray, n_fits: int) -> PriceMetrics:
    errors = predicted - actual
    return PriceMetrics(
        mae=round(float(np.mean(np.abs(errors))), 2),
        rmse=round(float(np.sqrt(np.mean(errors**2))), 2),
        mape=round(float(np.mean(np.abs(errors / actual))), 4),
        n_observations=len(actual),
        n_fits=n_fits,
    )


def cross_validate(
    frame: pd.DataFrame,
    estimators: tuple[PriceEstimator, ...],
    *,
    splits: int = CV_SPLITS,
    repeats: int = CV_REPEATS,
) -> dict[str, PriceMetrics]:
    """Repeated k-fold cross-validation, pooling out-of-fold predictions.

    Repeated rather than single k-fold because with a few dozen rows a single split is
    mostly luck: which four hotels land in the test fold moves MAE more than the choice of
    model does. Averaging over `repeats` shuffles gives a number worth quoting.

    Every estimator sees exactly the same folds, so the comparison against the baselines is
    paired rather than approximate.
    """
    prepared = prepare(frame)
    folds = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=RANDOM_STATE)
    collected: dict[str, tuple[list[float], list[float]]] = {
        estimator.name: ([], []) for estimator in estimators
    }
    n_fits = 0

    for train_index, test_index in folds.split(prepared):
        train = prepared.iloc[train_index]
        test = prepared.iloc[test_index]
        n_fits += 1
        for estimator in estimators:
            estimator.fit(train)
            actual, predicted = collected[estimator.name]
            actual.extend(test[TARGET].to_numpy().tolist())
            predicted.extend(estimator.predict(test).tolist())

    return {
        name: _metrics(np.array(actual), np.array(predicted), n_fits)
        for name, (actual, predicted) in collected.items()
    }
