# Raw Probability Analysis — Chronological Test Set (Friday)

## Summary Statistics

| Statistic | Benign (n=400,583) | Malicious (n=220,788) |
|-----------|-----------------|--------------------|
| Mean | 0.002856 | 0.160097 |
| Median | 0.000805 | 0.202511 |
| Std | 0.023160 | 0.164587 |
| Min | 0.000504 | 0.000800 |
| Max | 0.992626 | 0.994229 |

## Quantile Distribution

| Quantile | Benign | Malicious |
|----------|--------|-----------|
| 1% | 0.000785 | 0.001202 |
| 5% | 0.000792 | 0.001202 |
| 25% | 0.000805 | 0.001228 |
| 50% | 0.000805 | 0.202511 |
| 75% | 0.000953 | 0.218625 |
| 95% | 0.003603 | 0.426376 |
| 99% | 0.039366 | 0.761127 |

## Why Does Threshold 0.5 Fail?

Only 2.90% of malicious samples score ≥0.5. The model compresses attack probabilities: malicious median = 0.202511, 95th percentile = 0.426376. The classifier's probability surface has shifted under temporal distribution change.

- Fraction of benign traffic scoring ≥ 0.5: **0.06%**
- Fraction of malicious traffic scoring ≥ 0.5: **2.90%**

## Class Separation

Despite miscalibration, the ranking structure is preserved. Benign 99th pct = 0.039366 vs Malicious 25th pct = 0.001228. This confirms the model can still discriminate, but its probability estimates are not well-calibrated.

- Overlap region: [0.000800, 0.992626]
- Meaningful separation at quantile level: **True**
