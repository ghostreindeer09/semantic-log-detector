# The Evaluation Journey: From Lab to Production Reality

This document outlines the systematic, five-stage evaluation path undertaken to stress-test the LightGBM network intrusion detection system (IDS) on the CIC-IDS2017 dataset. Each stage reveals how standard machine learning assumptions break down when deployed against adversarial network traffic.

---

## 📈 Stage 1 — Random Stratified Split (The Illusion of Perfection)

In the initial development phase, the model was evaluated using standard cross-validation practices.

### Split Methodology
- **Cleaned Dataset**: 2,522,362 network flows (following cleaning and deduplication).
- **Split**: 70% Train, 15% Validation, 15% Test.
- **Stratification**: Preserved class ratios within each split.

### Empirical Results (Binary & Multiclass)
- **Binary F1**: `0.9974`
- **ROC-AUC**: `0.99998`
- **Macro-F1 (Multiclass)**: `0.9504`
- **False Positive Rate (FPR)**: `0.10%`
- **Detection Rate (Recall)**: `99.97%` (only 17 attacks missed out of 63,882)

### The Interpretation
The model appeared production-ready. The metric dashboard suggested near-perfect discrimination between benign and malicious flows and high granularity across individual attack categories.

---

## 📅 Stage 2 — Chronological Split Validation (The Production Crash)

To test the model's performance under realistic deployment conditions, we structured the evaluation temporally.

### Split Methodology
CIC-IDS2017 network traffic was captured across 5 weekdays, each featuring distinct attack categories:
- **Train (Mon–Wed)**: Benign baseline + Brute Force (Patator) + DoS (Hulk, GoldenEye, etc.).
- **Validation (Thu)**: Web Attacks + Infiltration.
- **Test (Fri)**: Botnets + PortScans + DDoS (LOIC volumetric attacks).

This chronological configuration tests the model's ability to detect unseen, future attacks based on features learned from past traffic.

### Results at Default Threshold (0.5)
- **F1 Score**: `0.0563`
- **Recall**: `2.90%` (missed 214,388 out of 220,788 attack flows)
- **Precision**: `96.41%`
- **FPR**: `0.06%`

### Root Cause: Temporal Leakage & Overfitting
In the random split, flows from the same 30-minute attack burst were distributed across train, validation, and test sets. The model memorized specific statistical fingerprints (identical IP addresses, ports, and inter-arrival times) instead of learning generalized features. When evaluated chronologically, this same-burst leakage was eliminated, exposing the model's complete inability to classify Friday's attacks at the default operating threshold.

---

## 🎚 Stage 3 — Threshold Sensitivity Study (Re-establishing Discrimination)

The second stage showed that while the binary F1 score collapsed, the model's ranking ability (**ROC-AUC = 0.9289**) remained strong. This indicated that the model could still rank attack flows higher than benign ones, but its decision boundary was misaligned.

### Empirical Sweep Results
We swept the decision threshold across 11 configurations from `0.0001` to `0.5`:

| Threshold | Accuracy | Precision | Recall | F1 | FPR | FNR |
|-----------|----------|-----------|--------|------|------|------|
| 0.0001 | 0.3553 | 0.3553 | 1.0000 | 0.5243 | 1.0000 | 0.0000 |
| **0.001 (Max F1)** | **0.8501** | **0.7042** | **0.9967** | **0.8253** | **0.2307** | **0.0033** |
| 0.005 | 0.8363 | 0.9136 | 0.5957 | 0.7212 | 0.0310 | 0.4043 |
| 0.05 | 0.8469 | 0.9726 | 0.5856 | 0.7310 | 0.0091 | 0.4144 |
| 0.5 (Default) | 0.6546 | 0.9641 | 0.0290 | 0.0563 | 0.0006 | 0.9710 |

### Key Takeaway
Recalibrating the decision threshold to **0.0012** recovered F1 to **0.8599** and recall to **99.56%**. However, this shift came at a high operational cost: the False Positive Rate increased from `0.06%` to `17.64%` (70,662 false alarms).

---

## 🎯 Stage 4 — Probability Calibration (The Shift-Within-Shift Failure)

To align the model's confidence estimates with true probabilities, we trained post-hoc calibrators on the Thursday validation set and evaluated them on Friday test data.

### Calibration Algorithms Tested
- **Platt Scaling** (Logistic Calibration)
- **Isotonic Regression** (Non-parametric Calibration)

### Empirical Calibration Metrics

| Method | Brier Score ↓ | ECE ↓ | Best F1 | ROC-AUC | Best Threshold |
|--------|--------------|-------|---------|---------|----------------|
| **Original LightGBM** | 0.2606 | 0.2968 | 0.8599 | 0.9289 | 0.0012 |
| **Platt Scaling** | 0.3297 | 0.3369 | 0.8599 | 0.9289 | 0.0044 |
| **Isotonic Regression** | 0.2012 | 0.2500 | 0.7385 | 0.9147 | 0.0002 |

### Why Calibration Failed
1. **Platt Scaling degraded reliability**: The Brier score increased from `0.26` to `0.33`. The sigmoid scaling function learned on Thursday's sparse 0.5% Web Attack/Infiltration distribution did not generalize to Friday's volumetric DDoS traffic.
2. **Isotonic Regression damaged discrimination**: While it slightly improved Brier and ECE scores, the step-function mapping overfit the sparse validation set, introducing non-monotonic artifacts that degraded ROC-AUC from `0.9289` to `0.9147` and F1 from `0.8599` to `0.7385`.
3. **Shift-Within-Shift**: Standard calibration assumes validation and test distributions are identical. Because the temporal distribution shifted between Thursday (Web Attacks) and Friday (DDoS/Botnets), the calibrator learned an incorrect mapping.

---

## 🎛 Stage 5 — Drift-Aware Adaptive Thresholding (The Normalization Trap)

To handle temporal drift without retraining, we implemented a streaming simulator that adjusts the decision threshold dynamically using sliding window statistics.

### Adaptive Strategies Evaluated
- **Quantile-Based Adaptation**: Set threshold at the $q$-th percentile of recent score windows.
- **Drift-Triggered Updates**: Perform KS tests comparing reference vs. current score distributions, triggering threshold resets upon shift.
- **Operational Guardrails**: Bounded step changes, smoothing ($\alpha=0.3$), and anomaly-based freeze locks (freezes threshold if anomaly rate exceeds 15%).

### Empirical Streaming Results (Friday Test)

| Strategy | Precision | Recall | F1 | FPR | Alerts | Threshold Range |
|----------|-----------|--------|------|------|--------|-----------------|
| **Fixed 0.5** | 0.9641 | 0.0290 | 0.0563 | 0.0006 | 6,638 | Fixed [0.500] |
| **Fixed 0.001** | 0.7042 | 0.9967 | 0.8253 | 0.2307 | 312,485 | Fixed [0.001] |
| **Q99_W5K** | 0.7985 | 0.0833 | 0.1509 | 0.0116 | 23,031 | [0.001, 0.500] |
| **Drift_Q995** | 0.7652 | 0.0421 | 0.0798 | 0.0071 | 12,151 | [0.001, 0.500] |

### The Unsupervised Normalization Collapse
While the adaptive strategies successfully reduced false alerts (e.g., `Q99_W5K` achieved a **92.6% alert reduction** compared to Fixed 0.001), they suffered a catastrophic drop in recall:
- `Q99_W5K` recall collapsed to **8.33%** (down from 99.67%).
- `Drift_Q995` recall collapsed to **4.21%**.

### Why Naive Adaptation Fails
During an active volumetric attack (such as DDoS), the influx of attack flows pushes the prediction score distribution upward. Without labels, the quantile-based adaptive controller interprets this upward shift as "the new normal" baseline traffic. The controller raises the decision threshold to maintain the target percentile, adapting *away* from the threat. 

Even with incident-aware freezing (which locked the threshold during 12.4% of the Friday evaluation), the sheer volume and duration of Friday's attacks bypassed the guards, demonstrating that **naive unsupervised threshold adaptation can normalize malicious behavior.**
