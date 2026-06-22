# Deployment Lessons & Operational Recommendations

This document translates the empirical findings of the chronological validation and calibration studies into concrete machine learning and detection engineering guidelines for security operation centers (SOC).

---

## ⚠️ Core ML Engineering Takeaways

### 1. Offline Metrics Are Misleading
Standard random train-test splits on intrusion datasets produce highly optimistic metrics (e.g., F1 = 99.7%, FPR = 0.1%). In security applications, random splits lead to **same-burst leakage**, where individual packets or flows from the same attack session are shared between train and test. This tests pattern memorization rather than generalization, hiding major threshold issues.
- **Guideline**: Always evaluate security models using a chronological day-based split that groups entire capture days or attack campaigns together.

### 2. High ROC-AUC Does Not Guarantee a Usable Threshold
Our model achieved an ROC-AUC of `0.9289` on chronological evaluation, meaning it retained high discriminative capacity. However, at the default threshold of `0.5`, recall collapsed to `2.9%`. 
- **Guideline**: Never assume a default threshold of 0.5 is appropriate for production deployment. The ROC-AUC represents *ranking quality* across all potential thresholds, but operational performance depends on selecting a single decision boundary.

### 3. The Shift-within-Shift Calibration Failure
Post-hoc calibration methods (e.g., Platt Scaling and Isotonic Regression) assume that the calibration training set (validation split) and the deployment set (test split) share the same distribution. Under temporal drift:
- The base model's predictions shift.
- The calibrator's features shift, rendering the calibration mapping obsolete.
- Platt scaling learned on Thursday (Web Attacks) worsened Brier scores on Friday (DDoS/Botnets), while Isotonic regression overfit and damaged the ranking order.
- **Guideline**: Simple threshold tuning on chronological validation data is more robust than post-hoc probability calibration under severe concept drift.

### 4. Naive Unsupervised Adaptation Can Normalize Attacks
Adjusting the threshold dynamically using sliding window quantiles (to control the false alert rate) introduces a severe vulnerability. During an active attack, the score distribution shifts upward. The quantile-based controller interprets this shift as "normal" background noise and raises the threshold, effectively adapting *away* from the threat.
- **Guideline**: Never deploy unsupervised, label-free adaptive thresholding without strict safeguards, such as anomaly-rate freezes or asymmetric step boundaries.

---

## 🛠 SOC Operational Recommendations

To deploy this model successfully, we recommend the following structural adjustments to the detection pipeline:

### 1. Separate Binary Detection from Multi-class Classification
The chronological study showed that the binary classifier generalized partially to unseen attacks (achieving F1 = 0.86 at threshold 0.0012), while the multiclass model collapsed to **0% detection** on unseen categories (classifying botnets and DDoS as benign).
- **Architecture**:
  - Use the binary LightGBM model at a low, calibrated threshold as the primary alert generator.
  - Route triggered alerts to a downstream system (rules, Sentence-BERT, or human analyst) for categorization. Do not rely on multiclass classification models for threat labeling on novel traffic.

### 2. Implement Calibrated Tiered Alerting
Rather than a single binary threshold, configure a tiered alerting structure to optimize analyst triage:

```
Score Region       Priority     Action
────────────────────────────────────────────────────────────────────────────
p >= 0.05          High         Auto-block traffic; high-priority paging
0.0012 <= p < 0.05 Medium       Route to SOC analyst triage queue
p < 0.0012         None         Log for offline batch analysis & monitoring
```

*Rationale*: High-scoring alerts ($p \ge 0.05$) achieve high precision ($\ge 97\%$). Recalibrated alerts ($0.0012 \le p < 0.05$) preserve recall but carry higher false alarms, making them ideal for analyst verification.

### 3. Deploy a Hybrid Supervised-Unsupervised Pipeline
Since supervised classifiers cannot generalize to completely novel feature spaces, combine the flow-based LightGBM model with unsupervised anomaly detectors:
- Use the **Isolation Forest** to identify volumetric rate anomalies that the LightGBM model has not seen.
- Use **Sentence-BERT** to analyze unstructured log messages semantically, providing metadata that flow features miss.

### 4. Continuous Feature-Level Drift Monitoring
Instead of adapting decision thresholds based on prediction scores alone, monitor feature distributions directly in production:
- Establish a baseline using training statistics (`drift_baseline.json`).
- Calculate the Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics over sliding windows of incoming features.
- If feature drift exceeds a threshold, alert the engineering team to retrain the model on recent labeled data, rather than attempting automated threshold adaptation.
