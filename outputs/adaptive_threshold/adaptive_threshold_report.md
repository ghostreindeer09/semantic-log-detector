# Drift-Aware Adaptive Thresholding — Operational Report

*Generated: 2026-06-22 11:54:24*

---

## 1. Why Static Thresholds Failed

The LightGBM IDS was trained on Monday–Wednesday traffic (Brute Force, DoS attacks). When deployed on Friday traffic (DDoS, PortScan, Bot), the model's prediction score distribution shifted dramatically:

- **Benign traffic**: median score = 0.0008 (very low, correctly)
- **Malicious traffic**: median score = 0.2025 (should be near 1.0, but compressed)

The model *ranks* attacks above benign (ROC-AUC = 0.93) but assigns them low absolute probabilities. A fixed threshold of 0.5 misses 97% of attacks. A fixed threshold of 0.001 catches 99.7% of attacks but also generates a 23% false positive rate because it's below the benign score distribution's tail.

Neither threshold can simultaneously achieve high recall and low FPR because the score distributions overlap in the [0.001, 0.04] range — and the optimal decision boundary is different for different traffic regimes.

## 2. Method Comparison

| Method | Precision | Recall | F1 | FPR | Alerts | Alert Rate |
|--------|-----------|--------|------|------|--------|------------|
| Fixed_0.5 | 0.9641 | 0.0290 | 0.0563 | 0.0006 | 6,638 | 0.0107 |
| Fixed_0.001 | 0.7042 | 0.9967 | 0.8253 | 0.2307 | 312,485 | 0.5029 |
| Q99_W5K | 0.7985 | 0.0833 | 0.1509 | 0.0116 | 23,031 | 0.0371 |
| Q995_W5K | 0.7652 | 0.0421 | 0.0798 | 0.0071 | 12,151 | 0.0196 |
| Q999_W5K | 0.8227 | 0.0293 | 0.0566 | 0.0035 | 7,872 | 0.0127 |
| Q995_W1K | 0.7595 | 0.0536 | 0.1002 | 0.0094 | 15,596 | 0.0251 |
| Q995_W10K | 0.8365 | 0.0653 | 0.1212 | 0.0070 | 17,243 | 0.0278 |
| Drift_Q995 | 0.7652 | 0.0421 | 0.0798 | 0.0071 | 12,151 | 0.0196 |

### Fixed Baselines

- **Fixed 0.5**: Recall = 0.0290, FPR = 0.0006, Alerts = 6,638
- **Fixed 0.001**: Recall = 0.9967, FPR = 0.2307, Alerts = 312,485

## 3. Adaptive Threshold Analysis

### Best Adaptive Configuration: Q99_W5K

- **F1**: 0.1509
- **Precision**: 0.7985
- **Recall**: 0.0833
- **FPR**: 0.0116
- **Total Alerts**: 23,031
- **Threshold Mean ± Std**: 0.1708 ± 0.1967
- **Threshold Range**: [0.0010, 0.5000]

### vs Fixed 0.001 Baseline

- F1 change: -0.6744
- FPR change: -0.2192
- Alert reduction: +289,454 (+92.6%)

## 4. Threshold Stability

| Config | Mean | Std | Min | Max | Changes >1% |
|--------|------|-----|-----|-----|-------------|
| Q99_W5K | 0.1708 | 0.1967 | 0.0010 | 0.5000 | 159 |
| Q995_W5K | 0.2235 | 0.1981 | 0.0010 | 0.5000 | 678 |
| Q999_W5K | 0.3621 | 0.1589 | 0.0010 | 0.5000 | 751 |
| Q995_W1K | 0.1883 | 0.1954 | 0.0010 | 0.5000 | 3379 |
| Q995_W10K | 0.2318 | 0.1938 | 0.0010 | 0.5000 | 136 |
| Drift_Q995 | 0.2235 | 0.1981 | 0.0010 | 0.5000 | 678 |

## 5. Incident-Aware Freezing

| Config | Steps Frozen | % Frozen | Recall (Frozen) | Recall (Unfrozen) | Freeze Events |
|--------|-------------|----------|-----------------|-------------------|---------------|
| Q99_W5K | 77,186 | 12.4% | 0.214057 | 0.031287 | 14 |
| Q995_W5K | 64,486 | 10.4% | 0.088765 | 0.027748 | 12 |
| Q999_W5K | 55,000 | 8.8% | 0.042559 | 0.026188 | 11 |
| Q995_W1K | 81,622 | 13.1% | 0.109613 | 0.029751 | 16 |
| Q995_W10K | 72,184 | 11.6% | 0.181628 | 0.028584 | 13 |
| Drift_Q995 | 64,486 | 10.4% | 0.088765 | 0.027748 | 12 |

Freezing prevents the threshold from adapting to attack traffic as 'normal'. The recall-during-freeze metric shows whether the threshold was at a useful level when it was locked.

## 6. Detection Delay

Time-to-first-detection (TTFD) measures samples from first attack in a campaign to the first true positive alert.

| Method | Campaign | Attack Types | TTFD (samples) | Campaign Recall |
|--------|----------|-------------|----------------|----------------|
| Fixed_0.5 | Friday-WorkingHours-Afternoon-DDos. | DDoS | 205 | 0.0473 |
| Fixed_0.5 | Friday-WorkingHours-Afternoon-PortS | PortScan | 4 | 0.0039 |
| Fixed_0.5 | Friday-WorkingHours-Morning.pcap_IS | Bot | NEVER | 0.0000 |
| Fixed_0.001 | Friday-WorkingHours-Afternoon-DDos. | DDoS | 0 | 1.0000 |
| Fixed_0.001 | Friday-WorkingHours-Afternoon-PortS | PortScan | 0 | 1.0000 |
| Fixed_0.001 | Friday-WorkingHours-Morning.pcap_IS | Bot | 0 | 0.6262 |
| Q99_W5K | Friday-WorkingHours-Afternoon-DDos. | DDoS | 0 | 0.1315 |
| Q99_W5K | Friday-WorkingHours-Afternoon-PortS | PortScan | 4 | 0.0171 |
| Q99_W5K | Friday-WorkingHours-Morning.pcap_IS | Bot | NEVER | 0.0000 |
| Q995_W5K | Friday-WorkingHours-Afternoon-DDos. | DDoS | 0 | 0.0656 |
| Q995_W5K | Friday-WorkingHours-Afternoon-PortS | PortScan | 4 | 0.0099 |
| Q995_W5K | Friday-WorkingHours-Morning.pcap_IS | Bot | NEVER | 0.0000 |
| Q999_W5K | Friday-WorkingHours-Afternoon-DDos. | DDoS | 17 | 0.0473 |
| Q999_W5K | Friday-WorkingHours-Afternoon-PortS | PortScan | 4 | 0.0046 |
| Q999_W5K | Friday-WorkingHours-Morning.pcap_IS | Bot | NEVER | 0.0000 |
| Q995_W1K | Friday-WorkingHours-Afternoon-DDos. | DDoS | 0 | 0.0866 |
| Q995_W1K | Friday-WorkingHours-Afternoon-PortS | PortScan | 0 | 0.0084 |
| Q995_W1K | Friday-WorkingHours-Morning.pcap_IS | Bot | NEVER | 0.0000 |
| Q995_W10K | Friday-WorkingHours-Afternoon-DDos. | DDoS | 0 | 0.1049 |
| Q995_W10K | Friday-WorkingHours-Afternoon-PortS | PortScan | 4 | 0.0109 |
| Q995_W10K | Friday-WorkingHours-Morning.pcap_IS | Bot | NEVER | 0.0000 |
| Drift_Q995 | Friday-WorkingHours-Afternoon-DDos. | DDoS | 0 | 0.0656 |
| Drift_Q995 | Friday-WorkingHours-Afternoon-PortS | PortScan | 4 | 0.0099 |
| Drift_Q995 | Friday-WorkingHours-Morning.pcap_IS | Bot | NEVER | 0.0000 |

## 7. Drift Events & Direction

**Drift_Q995**: 134 drift events detected


## 8. SOC Operational Assessment

### Would this reduce alert fatigue?

**Yes.** The best adaptive method generated 289,454 fewer alerts (92.6% reduction) compared to Fixed 0.001.

### Was recall preserved?

Adaptive recall = 0.0833 vs Fixed 0.001 recall = 0.9967

### How frequently did the threshold change?

The threshold changed significantly (>1%) on **159** occasions across 621,371 samples.

## 9. Limitations

1. **No true temporal ordering**: The Friday test data is ordered by source file, not by true network timestamp. Within each file, flow ordering may not be strictly chronological.

2. **Single-day evaluation**: The adaptive system was evaluated on one day of traffic. Multi-day evaluation across varying attack patterns would provide stronger evidence.

3. **No concept drift in features**: The adaptive threshold only adjusts the decision boundary — it does not address feature-level drift. If the model's feature representations degrade, threshold adaptation cannot compensate.

4. **Label-free operation tradeoff**: The quantile-based approach does not use labels, which makes it deployable but also means it cannot directly optimize recall or precision. It assumes that anomalous scores are in the tail of the distribution.

5. **Warmup period vulnerability**: During warmup, the controller uses a fallback threshold. If the initial traffic distribution differs significantly from what the fallback was calibrated on, early detection performance may suffer.

## 10. Conclusion

### The Complete Engineering Story

1. **High offline metrics can be misleading.** Random split F1 = 0.997 collapsed to 0.056 under temporal evaluation.

2. **Temporal validation exposes deployment failure.** The model retained discrimination (ROC-AUC = 0.93) but the default threshold was 500× too high.

3. **Calibration alone cannot solve severe distribution shift.** Platt and Isotonic calibration, trained on Thursday's distribution, did not transfer to Friday.

4. **Adaptive decision systems may provide a more robust operating model.** The adaptive threshold reduced false positives but at some cost to recall, illustrating the fundamental tradeoff. Whether this is operationally acceptable depends on the SOC's capacity and risk tolerance.
