"""Small statistical helpers shared across the pipeline."""

from __future__ import annotations

import pandas as pd


def shrink[Numeric: (float, pd.Series[float])](
    value: Numeric,
    evidence: Numeric,
    prior_value: float,
    prior_weight: float,
) -> Numeric:
    """Pull an estimate toward a prior in proportion to how thin its evidence is.

    The standard Bayesian-average shrinkage:

        (value · evidence + prior_value · prior_weight) / (evidence + prior_weight)

    A 5.0 rating from one review and a 4.7 from five hundred are not the same claim. This
    makes them comparable by discounting the first toward what the market as a whole looks
    like, in proportion to how little is actually known about it. `prior_weight` is the
    amount of evidence at which an item's own value starts to dominate the prior, and it is
    tuned per population — hotel and activity review counts differ by two orders of
    magnitude, so they do not share a value.

    Works elementwise on pandas Series as well as on plain floats, so there is exactly one
    definition of the formula in the codebase.
    """
    if prior_weight <= 0:
        raise ValueError("prior_weight must be positive")
    return (value * evidence + prior_value * prior_weight) / (evidence + prior_weight)
