# AeroTwin Statistical Protocol (Frozen)

**Protocol version:** 1.0
**Status:** Frozen — changes require a documented protocol revision and a version bump.

This document is the single authoritative reference for the statistical
inference methodology used across all AeroTwin experiments: Level 1 flight-level
evaluation, Level 2 leave-one-type-out (LOTO) evaluation, and external /
cross-dataset generalization. The machine-readable source of truth is
`physics/statistical_protocol.py`; this file states the *intent* and the
*reasons* behind each choice so the protocol cannot silently drift between
notebooks.

---

## 1. Frozen constants

| Constant | Value | Meaning |
|---|---|---|
| `RANDOM_STATE` | `42` | Seed for every split, bootstrap, and randomized model. |
| `TEST_SIZE` | `0.2` | Held-out fraction for strict flight-level (Level 1) evaluation. |
| `N_BOOTSTRAP` | `10_000` | Bootstrap resamples for flight-clustered MAE-difference inference. |
| `ALPHA` | `0.05` | Nominal significance level (one-sided at `ALPHA`). |
| `CI_LEVEL` | `0.95` | Bootstrap confidence level → 2.5 / 97.5 percentiles. |
| `REPLICATION_P_THRESHOLD` | `0.95` | P(new better) needed to call a finding *replicated* on one dataset. |

---

## 2. Split protocol (no leakage)

- **Level 1 (unseen flights):** split by `flight_id` (80/20, `random_state=42`).
  Zero flights shared between train and test. Intervals from the same flight
  never cross the train/test boundary.
- **Level 2 (unseen aircraft families):** leave-one-type-out. All flights of one
  ICAO type are held out of training and scored only on that type.

---

## 3. Inference unit must match the claim

The resampling unit is chosen by the scientific claim, **never** by convenience:

| Claim | Correct resampling unit (`unit`) |
|---|---|
| Unseen-flight generalization | `flight_level` (resample held-out flights) |
| Unseen-aircraft-family generalization (LOTO) | `type_level` (resample the 12 held-out types) |
| LOTO paired Flow-vs-Direct comparison | `hierarchical_type_to_flight` (type→flight) |
| Interval-level Wilcoxon | Supplementary only, when flights are correlated |

`physics/statistical_protocol.frozen_significance(...)` rejects any `unit` that
is not in `ALLOWED_INFERENCE_UNITS`.

---

## 4. Primary inference: flight-clustered bootstrap

- Resample **test flights** with replacement (not intervals).
- For each resample compute the difference in mean absolute error
  `MAE(new) − MAE(baseline)` over the resampled flights.
- Report the 95% bootstrap CI on ΔMAE.
- The point estimate is the flight-clustered MAE difference; the bootstrap
  distribution is the basis for the confidence interval and the one-sided
  `P(new better)`.

---

## 5. Interpretation policy (frozen)

This rule is implemented once in `classify_significance` and must not be
re-implemented per notebook:

- **New significantly better** ⇔ `ci_upper < 0` **and** `P(new worse) < ALPHA`.
  (`P(new worse)` is the fraction of bootstrap resamples where
  `MAE(new) > MAE(baseline)`.)
- **Baseline significantly better** ⇔ `ci_lower > 0`.
- **Otherwise: "No significant evidence."** Do **not** claim or imply an effect
  when the CI crosses zero.

Confirmatory claims require the CI to exclude zero. Exploratory analyses
(physical-transfer-distance correlations, `n = 12` type samples) are reported as
such and are explicitly flagged as influence-sensitive; they do not support
primary claims.

---

## 6. Cross-dataset / external replication

A qualitative finding (e.g. Flow+Energy beats Direct) is considered **replicated
on a single external dataset** when:

- `MAE(new) < MAE(baseline)` **and**
- `P(new better) >= REPLICATION_P_THRESHOLD` (`0.95`).

Aggregating across datasets yields a meta-verdict: replicated on all / partial /
none. Negative or non-replicated results are reported explicitly, not hidden.

---

## 7. Reporting rules

1. Bootstrap CI is the primary evidence; never claim significance when the CI
   crosses zero.
2. Match the resampling unit to the claim (§3).
3. Distinguish confirmatory vs exploratory; mark small-`n` analyses as
   influence-sensitive.
4. Negative results are scientifically valuable and are reported.
5. Competition proximity (e.g. PRC leaderboard RMSE) is **not** external
   validation and must not be presented as such.

---

## 8. Change control

Any edit to a constant in §1, the unit whitelist in §3, or the policy in §5 is a
**protocol change**. It requires:

- an updated version in `PROTOCOL_VERSION`,
- a corresponding edit to this document, and
- a note in the project status report.
