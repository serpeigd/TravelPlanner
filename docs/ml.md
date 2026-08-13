# The machine learning component

## What it models, and why the prediction is not the product

The model regresses a property's **nightly price** on the property's own attributes. The
prediction is worthless on its own — nobody needs an estimate of a price that is printed on
the page. The **residual** is the product:

```
value_for_money  ←  observed price  vs.  price predicted from the property's own features
```

A room priced well below what its stars, amenities, location and reviews predict is good
value *for what it is*. One priced well above is not. This is a **hedonic regression** in the
economics sense — the same construction used for house-price indices and CPI quality
adjustment — and it answers a question the traveller actually has and cannot answer by
looking at a price list.

## Why price and not learning-to-rank

The obvious ML move in a recommender is learning-to-rank. It needs relevance labels — clicks,
bookings, dwell time — and we have none. Fabricating them would produce a model that learns
the heuristic used to fabricate them, dressed up as evidence. Price is the one target that
genuinely exists in this data. See [ADR-003](decisions.md).

## Reproducing the numbers

```bash
python -m travel_intel.ml.report
```

Deterministic: same snapshot, same folds, same output.

## Results

30 properties, €70–€647 per night, median €94. Repeated 5-fold cross-validation, 5 repeats,
25 fits per estimator, every estimator on the same folds (paired, not approximate):

| Estimator | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| baseline: global median | €80.98 | €154.23 | 31.4 % |
| baseline: district median | €73.32 | €142.43 | 26.7 % |
| **hedonic ridge (log price)** | **€40.85** | **€77.14** | **20.9 %** |

The model cuts mean absolute error by **44 % against the district median** — the honest
competitor, since location explains a lot of price in any city and a grouped median needs no
model at all. A regression that could not beat it would not have earned its place, and the
finding would have been reported as such.

**These are out-of-sample numbers.** The residuals used for ranking are in-sample by design
(see below); generalisation is measured separately, and this table is that measurement.

## What the model learned

Standardised coefficients, log-price units:

| Feature | Coefficient |
|---|---:|
| `stars` | +0.2914 |
| `n_amenities` | +0.2251 |
| `distance_km` | −0.1722 |
| `log_review_count` | −0.0734 |
| `rating` | −0.0141 |

Stars dominate, amenities follow, distance from Tokyo Station pushes price down — all as
expected, which is a sanity check rather than a discovery.

The interesting result is the last row. **Guest rating barely explains price at all.** Once
you know what a hotel *is*, how guests scored it adds almost nothing to what it *charges*.
That is a finding, not a bug, and it justifies a design decision made before the model
existed: `rating` and `value_for_money` are separate ranking factors because they carry
genuinely different information. If the rating coefficient had been large, the two factors
would have been double-counting the same signal.

The mildly negative `log_review_count` is also readable: the highest review counts in this
sample belong to large budget chains (one APA hotel has 23,178 reviews), so review volume
proxies for scale rather than prestige.

## Two decisions about honesty

**In-sample residuals, on purpose.** The model is fitted on the candidate set and predicts it
back. "Cheap for what it is" is a claim about a *market*, so the market it is measured
against should be the options actually on the table. Reporting cross-validated error
separately is what keeps this honest: the ranking uses hedonic residuals, the metrics
table reports generalisation, and the two are never conflated.

**Trained on the whole market, not the affordable subset.** Candidate generation drops
everything above the traveller's budget. Training on what survives would truncate the target:
the model would never see the expensive end and would systematically under-predict what a
good hotel costs, making every luxury property look like a bargain. There is a test for this.

## Modelling choices

- **Log target.** Prices span €70–€647 and are right-skewed. On the raw scale, errors on
  expensive hotels dominate the fit. In log space the model learns *proportional* differences,
  which is also how the residual is interpreted.
- **Ridge, not OLS.** Stars, amenity count and rating are correlated; regularisation keeps
  coefficients stable instead of letting collinearity throw them around.
- **Five features, not sixteen.** Eleven amenity indicators were available and are collapsed
  into `n_amenities`. At this sample size, one parameter per amenity fits noise and produces
  coefficients that look meaningful and are not.
- **Median imputation, inside the pipeline.** One property has no stars and no rating.
  Imputing a *feature* inside a model, with a strategy learned on the training fold only, is
  standard practice — and different in kind from imputing a *score*, which the ranking
  refuses to do because it would fabricate a judgement. See [docs/ranking.md](ranking.md).

## Limitations

1. **30 rows.** This is a demonstrative model with honest error bars, not a production
   estimator. Five features for thirty observations is already at the edge of what the data
   supports, which is why the feature set is deliberately small.
2. **One city, one date range, one party size.** Nothing here generalises to another
   destination or season. Seasonality, day-of-week and lead time are the obvious missing
   variables and are all absent from a single-snapshot capture.
3. **No temporal validation.** With one snapshot there is no time axis, so k-fold is the
   honest choice. With a real price history, the split would have to be temporal — random
   folds over time series leak the future into the past.
4. **Star ratings are partly estimated by the provider** (`stars_type` is
   `estimated_by_booking` for several properties). That noise sits in the strongest feature.

## What would make this a real model

Historical prices per property per date: seasonality, day-of-week, lead-time-to-arrival,
event calendars, competitor pricing in the same micro-area. That turns a cross-sectional
hedonic regression into a demand-and-price model, and turns "is this good value today" into
"should this traveller book now or wait" — the question a travel company actually monetises.
